import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.base import DGLError
from dgl.ops import edge_softmax
import dgl.function as fn


# 一个不执行任何操作的算子，只是将输入原封不动的返回
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

class MsgMalPredictor(nn.Module):
    def __init__(self, emb_dim, out_classes):
        super(MsgMalPredictor, self).__init__()
        self.src_fc = nn.Linear(emb_dim, emb_dim)
        self.dst_fc = nn.Linear(emb_dim, emb_dim)
        self.out_fc = nn.Linear(emb_dim, out_classes)

    def mal_pred(self, edges):
        src_hid = self.src_fc(edges.src['embedding'])
        dst_hid = self.dst_fc(edges.dst['embedding'])
        edge_embs = F.relu(src_hid+dst_hid)
        score = self.out_fc(edge_embs)
        return {'score': score, 'emb': edge_embs}

    def forward(self, x, g):
        # Local Scope?
        g.ndata['embedding'] = x
        g.apply_edges(self.mal_pred)
        escore = g.edata['score']
        embs = g.edata['emb']
        return escore, embs

class ContrastModule(nn.Module):
    def __init__(self, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(ContrastModule, self).__init__()
        self.tau: float = tau

        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

# 一个用于链接预测的模块，使用消息传递的方式来预测正样本子图和负样本子图之间的链接
# 这个模块的目的是从子图中学习链接预测任务，其中正样本子图和负样本子图都包含节点特征以及它们之间的链接信息。
# 通过消息传递和神经网络的组合，模块试图学习如何从节点特征中预测链接的存在或缺失。
class MsgLinkPredictor(nn.Module):
    """Predict Pair wise link from pos subg and neg subg
    use message passing.

    Use Two layer MLP on edge to predict the link probability

    Parameters
    ----------
    embed_dim : int
        dimension of each each feature's embedding

    Example
    ----------
    >>> linkpred = MsgLinkPredictor(10)
    >>> pos_g = dgl.graph(([0,1,2,3,4],[1,2,3,4,0]))
    >>> neg_g = dgl.graph(([0,1,2,3,4],[2,1,4,3,0]))
    >>> x = torch.ones(5,10)
    >>> linkpred(x,pos_g,neg_g)
    (tensor([[0.0902],
         [0.0902],
         [0.0902],
         [0.0902],
         [0.0902]], grad_fn=<AddmmBackward>),
    tensor([[0.0902],
         [0.0902],
         [0.0902],
         [0.0902],
         [0.0902]], grad_fn=<AddmmBackward>))
    """

    def __init__(self, emb_dim):
        super(MsgLinkPredictor, self).__init__()
        self.src_fc = nn.Linear(emb_dim, emb_dim)
        self.dst_fc = nn.Linear(emb_dim, emb_dim)
        self.out_fc = nn.Linear(emb_dim, 1)

    def link_pred(self, edges):
        src_hid = self.src_fc(edges.src['embedding'])
        dst_hid = self.dst_fc(edges.dst['embedding'])
        score = F.relu(src_hid+dst_hid)
        score = self.out_fc(score)
        return {'score': score}

    def forward(self, x, pos_g, neg_g):
        # Local Scope?
        pos_g.ndata['embedding'] = x
        neg_g.ndata['embedding'] = x

        pos_g.apply_edges(self.link_pred)
        neg_g.apply_edges(self.link_pred)

        pos_escore = pos_g.edata['score']
        neg_escore = neg_g.edata['score']
        return pos_escore, neg_escore

# 用于编码时间信息的PyTorch模块，他使用有限傅立叶级数来编码两个事件之间的时间差异。该模块的目的是将
# 时间信息映射成一组特征，以便在时间序列数据中捕获不同的时间尺度信息。
class TimeEncode(nn.Module):
    """Use finite fourier series with different phase and frequency to encode
    time different between two event

    ..math::
        \Phi(t) = [\cos(\omega_0t+\psi_0),\cos(\omega_1t+\psi_1),...,\cos(\omega_nt+\psi_n)] 

    Parameter
    ----------
    dimension : int
        Length of the fourier series. The longer it is , 
        the more timescale information it can capture

    Example
    ----------
    >>> tecd = TimeEncode(10)
    >>> t = torch.tensor([[1]])
    >>> tecd(t)
    tensor([[[0.5403, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]], dtype=torch.float64, grad_fn=<CosBackward>)
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .double().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).double())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output

class MemoryModule(nn.Module):
    """Memory module as well as update interface

    The memory module stores both historical representation in last_update_t

    Parameters
    ----------
    n_node : int
        number of node of the entire graph

    hidden_dim : int
        dimension of memory of each node

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py;
                     examples/pytorch/tgn/train.py 

    """

    def __init__(self, n_node, hidden_dim):
        super(MemoryModule, self).__init__()
        self.n_node = n_node
        self.hidden_dim = hidden_dim
        self.reset_memory()

    def reset_memory(self):
        self.last_update_t = nn.Parameter(torch.zeros(
            self.n_node).float(), requires_grad=False)
        self.memory = nn.Parameter(torch.zeros(
            (self.n_node, self.hidden_dim)).float(), requires_grad=False)

    def backup_memory(self):
        """
        Return a deep copy of memory state and last_update_t
        For test new node, since new node need to use memory upto validation set
        After validation, memory need to be backed up before run test set without new node
        so finally, we can use backup memory to update the new node test set
        """
        return self.memory.clone(), self.last_update_t.clone()

    def restore_memory(self, memory_backup):
        """Restore the memory from validation set

        Parameters
        ----------
        memory_backup : (memory,last_update_t)
            restore memory based on input tuple
        """
        self.memory = memory_backup[0].clone()
        self.last_update_t = memory_backup[1].clone()

    # Which is used for attach to subgraph
    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    # When the memory need to be updated
    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def set_last_update_t(self, node_idxs, values):
        self.last_update_t[node_idxs] = values

    # For safety check
    def get_last_update(self, node_idxs):
        return self.last_update_t[node_idxs]

    def detach_memory(self):
        """
        Disconnect the memory from computation graph to prevent gradient be propagated multiple
        times
        """
        self.memory.detach_()


class MemoryOperation(nn.Module):
    """ Memory update using message passing manner, update memory based on positive
    pair graph of each batch with recurrent module GRU or RNN

    Message function
    ..math::
        m_i(t) = concat(memory_i(t^-),TimeEncode(t),v_i(t))

    v_i is node feature at current time stamp

    Aggregation function
    ..math::
        \bar{m}_i(t) = last(m_i(t_1),...,m_i(t_b))

    Update function
    ..math::
        memory_i(t) = GRU(\bar{m}_i(t),memory_i(t-1))

    Parameters
    ----------

    updater_type : str
        indicator string to specify updater

        'rnn' : use Vanilla RNN as updater

        'gru' : use GRU as updater

    memory : MemoryModule
        memory content for update

    e_feat_dim : int
        dimension of edge feature

    temporal_dim : int
        length of fourier series for time encoding

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py
    """

    def __init__(self, updater_type, memory, e_feat_dim, temporal_encoder):
        super(MemoryOperation, self).__init__()
        updater_dict = {'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.memory = memory
        memory_dim = self.memory.hidden_dim
        self.temporal_encoder = temporal_encoder
        # 消息的维度，由节点的记忆、时间编码、边特征和节点特征组成
        self.message_dim = memory_dim+memory_dim + \
            e_feat_dim+self.temporal_encoder.dimension
        self.updater = updater_dict[updater_type](input_size=self.message_dim,
                                                  hidden_size=memory_dim)
        self.memory = memory

    # Here assume g is a subgraph from each iteration
    def stick_feat_to_graph(self, g):
        # How can I ensure order of the node ID
        g.ndata['timestamp'] = self.memory.last_update_t[g.ndata[dgl.NID]]
        g.ndata['memory'] = self.memory.memory[g.ndata[dgl.NID]]

    # 讲道理源节点和目标节点都要更新，因此应该使用双向图
    def msg_fn_cat(self, edges):
        src_delta_time = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(src_delta_time.unsqueeze(
            dim=1)).view(len(edges.data['timestamp']), -1)
        # ToDo：使用其他消息函数
        ret = torch.cat([edges.src['memory'], edges.dst['memory'],
                         edges.data['feats'], time_encode], dim=1)
        return {'message': ret, 'timestamp': edges.data['timestamp']}

    # ToDo：使用其他聚合函数（RNN、Attention）
    # 这里使用的是最新消息，也可以采用平均消息
    def agg_last(self, nodes):
        timestamp, latest_idx = torch.max(nodes.mailbox['timestamp'], dim=1)
        ret = nodes.mailbox['message'].gather(1, latest_idx.repeat(
            self.message_dim).view(-1, 1, self.message_dim)).view(-1, self.message_dim)
        return {'message_bar': ret.reshape(-1, self.message_dim), 'timestamp': timestamp}

    def update_memory(self, nodes):
        # It should pass the feature through RNN
        ret = self.updater(
            nodes.data['message_bar'].float(), nodes.data['memory'].float())
        return {'memory': ret}

    def forward(self, g):
        self.stick_feat_to_graph(g)
        g.update_all(self.msg_fn_cat, self.agg_last, self.update_memory)
        return g

# 一个基于图注意力机制的图卷积层，用于处理图数据，可以接受节点特征和边特征，然后通过多头自注意力机制来更新节点特征。
class EdgeGATConv(nn.Module):
    '''Edge Graph attention compute the graph attention from node and edge feature then aggregate both node and
    edge feature.

    Parameter
    ==========
    node_feats : int
        number of node features

    edge_feats : int
        number of edge features

    out_feats : int
        number of output features

    num_heads : int
        number of heads in multihead attention

    # 节点特征的 dropout 比率（可选）
    feat_drop : float, optional
        drop out rate on the feature

    # 注意力权重的 dropout 比率（可选）
    attn_drop : float, optional
        drop out rate on the attention weight

    # LeakyReLU 函数的负斜率（可选）
    negative_slope : float, optional
        LeakyReLU angle of negative slope.

    # 是否使用残差连接（可选）
    residual : bool, optional
        whether use residual connection

    # 是否允许图中存在入度为零的节点（可选）
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

    '''

    def __init__(self,
                 node_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(EdgeGATConv, self).__init__()
        self._num_heads = num_heads
        self._node_feats = node_feats
        self._edge_feats = edge_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_node = nn.Linear(
            self._node_feats, self._out_feats*self._num_heads)
        self.fc_edge = nn.Linear(
            self._edge_feats, self._out_feats*self._num_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # 配置残差连接，并根据输入和输出特征的维度选择是否添加线性变换层
        self.residual = residual
        if residual:
            if self._node_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._node_feats, self._out_feats*self._num_heads, bias=False)
            else:
                self.res_fc = Identity()
        self.reset_parameters()
        self.activation = activation

    # 使用Xavier 初始化权重，这种初始化方法旨在确保网络在前向和反向传播时的梯度保持稳定，从而提高训练的稳定性和效率
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if self.residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    # message function
    def msg_fn(self, edges):
        ret = edges.data['a'].view(-1, self._num_heads,
                                   1)*edges.data['el_prime']
        return {'m': ret}


    def forward(self, graph, nfeat, efeat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            nfeat = self.feat_drop(nfeat)
            efeat = self.feat_drop(efeat)

            node_feat = self.fc_node(
                nfeat).view(-1, self._num_heads, self._out_feats)
            edge_feat = self.fc_edge(
                efeat).view(-1, self._num_heads, self._out_feats)

            el = (node_feat*self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (node_feat*self.attn_r).sum(dim=-1).unsqueeze(-1)
            ee = (edge_feat*self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.ndata['ft'] = node_feat
            graph.ndata['el'] = el
            graph.ndata['er'] = er
            graph.edata['ee'] = ee
            # el？ src or dst？
            graph.apply_edges(fn.u_add_e('el', 'ee', 'el_prime'))
            graph.apply_edges(fn.e_add_v('el_prime', 'er', 'e'))
            e = self.leaky_relu(graph.edata['e'])
            # 计算边的注意力权重
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.edata['efeat'] = edge_feat
            graph.update_all(self.msg_fn, fn.sum('m', 'ft'))
            rst = graph.ndata['ft']
            # print("rst.shape:", rst.shape)
            if self.residual:
                resval = self.res_fc(nfeat).view(
                    nfeat.shape[0], -1, self._out_feats)
                # print("resval.shape:", resval.shape)
                rst = rst + resval
                # print("rst.shape:", rst.shape)

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class TemporalEdgePreprocess(nn.Module):
    '''Preprocess layer, which finish time encoding and concatenate 
    the time encoding to edge feature.

    Parameter
    ==========
    edge_feats : int
        number of orginal edge feature

    temporal_encoder : torch.nn.Module
        time encoder model
    '''

    def __init__(self, edge_feats, temporal_encoder):
        super(TemporalEdgePreprocess, self).__init__()
        self.edge_feats = edge_feats
        self.temporal_encoder = temporal_encoder

    def edge_fn(self, edges):
        t0 = torch.zeros_like(edges.dst['timestamp'])
        time_diff = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(
            time_diff.unsqueeze(dim=1)).view(t0.shape[0], -1)
        edge_feat = torch.cat([edges.data['feats'], time_encode], dim=1)
        return {'efeat': edge_feat}

    def forward(self, graph):
        graph.apply_edges(self.edge_fn)
        efeat = graph.edata['efeat']
        return efeat

class TemporalTransformerConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_encoder,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False,
                 layers=1):
        '''Temporal Transformer model for TGN and TGAT

        Parameter
        ==========
        edge_feats : int
            number of edge features

        memory_feats : int
            dimension of memory vector

        temporal_encoder : torch.nn.Module
            compute fourier time encoding

        out_feats : int
            number of out features

        num_heads : int
            number of attention head

        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        '''
        super(TemporalTransformerConv, self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats = memory_feats
        self.temporal_encoder = temporal_encoder
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self.layers = layers

        self.preprocessor = TemporalEdgePreprocess(
            self._edge_feats, self.temporal_encoder)
        self.layer_list = nn.ModuleList()
        self.layer_list.append(EdgeGATConv(node_feats=self._memory_feats,
                                           edge_feats=self._edge_feats+self.temporal_encoder.dimension,
                                           out_feats=self._out_feats,
                                           num_heads=self._num_heads,
                                           feat_drop=0.6,
                                           attn_drop=0.6,
                                           residual=True,
                                           allow_zero_in_degree=allow_zero_in_degree))
        for i in range(self.layers-1):
            self.layer_list.append(EdgeGATConv(node_feats=self._out_feats*self._num_heads,
                                               edge_feats=self._edge_feats+self.temporal_encoder.dimension,
                                               out_feats=self._out_feats,
                                               num_heads=self._num_heads,
                                               feat_drop=0.6,
                                               attn_drop=0.6,
                                               residual=True,
                                               allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, graph, memory, ts):
        graph = graph.local_var()
        graph.ndata['timestamp'] = ts
        efeat = self.preprocessor(graph).float() # temporal node features vj (t).
        rst = memory # current memory sj(t)
        # 第i层的输入：它由节点在(i-1)层的表示、当前时间戳、邻域节点在各个时间戳的表示，以及每个形成节点的时序邻域边缘的交互特征组成。
        for i in range(self.layers-1):
            rst = self.layer_list[i](graph, rst, efeat).flatten(1)
        rst = self.layer_list[-1](graph, rst, efeat).mean(1)
        return rst
