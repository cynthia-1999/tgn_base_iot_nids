import copy
import torch.nn as nn
import dgl
from graph_model.modules import MemoryModule, MemoryOperation, MsgLinkPredictor, MsgMalPredictor, TemporalTransformerConv, TimeEncode


class TGN(nn.Module):
    def __init__(self,
                 edge_feat_dim,
                 out_classes,
                 memory_dim,
                 temporal_dim,
                 embedding_dim,
                 num_heads,
                 num_nodes,
                 n_neighbors=10,
                 memory_updater_type='gru',
                 layers=1, 
                 device='cpu'):
        super(TGN, self).__init__()
        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim
        self.temporal_dim = temporal_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.n_neighbors = n_neighbors
        self.memory_updater_type = memory_updater_type
        self.num_nodes = num_nodes
        self.layers = layers
        self.device = device

        self.temporal_encoder = TimeEncode(self.temporal_dim).to(device)

        self.memory = MemoryModule(self.num_nodes,
                                   self.memory_dim).to(device)

        self.memory_ops = MemoryOperation(self.memory_updater_type,
                                          self.memory,
                                          self.edge_feat_dim,
                                          self.temporal_encoder).to(device)

        self.embedding_attn = TemporalTransformerConv(self.edge_feat_dim,
                                                      self.memory_dim,
                                                      self.temporal_encoder,
                                                      self.embedding_dim,
                                                      self.num_heads,
                                                      layers=self.layers,
                                                      allow_zero_in_degree=True).to(device)

        self.msg_linkpredictor = MsgLinkPredictor(embedding_dim).to(device)
        self.msg_malpredictor = MsgMalPredictor(embedding_dim, out_classes).to(device)

    # def embed(self, postive_graph, negative_graph, blocks):
    #     emb_graph = blocks[0]
    #     print("self.memory.memory.device: ", self.memory.memory.device)
    #     print("emb_graph.device: ", emb_graph.device)
    #     emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID], :]
    #     emb_t = emb_graph.ndata['timestamp']
    #     # 过Temporal Transformer捕捉节点的时序特征
    #     embedding = self.embedding_attn(emb_graph, emb_memory, emb_t)
    #     emb2pred = dict(
    #         zip(emb_graph.ndata[dgl.NID].tolist(), emb_graph.nodes().tolist()))
    #     # Since postive graph and negative graph has same is mapping
    #     feat_id = [emb2pred[int(n)] for n in postive_graph.ndata[dgl.NID]]
    #     feat = embedding[feat_id]
    #     # 用MsgLinkPredictor预测节点连接的概率
    #     pred_pos, pred_neg = self.msg_linkpredictor(
    #         feat, postive_graph, negative_graph)
    #     return pred_pos, pred_neg
    def embed(self, graph, blocks):
        emb_graph = blocks[0]
        emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID], :]
        emb_t = emb_graph.ndata['timestamp']
        embedding = self.embedding_attn(emb_graph, emb_memory, emb_t)
        emb2pred = dict(
            zip(emb_graph.ndata[dgl.NID].tolist(), emb_graph.nodes().tolist()))
        # Since postive graph and negative graph has same is mapping
        # print("graph.ndata[dgl.NID]:", graph.ndata[dgl.NID])
        feat_id = [emb2pred[int(n)] for n in graph.ndata[dgl.NID]]
        feat = embedding[feat_id]
        # mal_pred = self.msg_malpredictor(
        #     feat, graph)
        # return mal_pred
        return feat

    def predict(self, graph, embeddings):
        mal_pred = self.msg_malpredictor(
            embeddings, graph)
        return mal_pred

    def update_memory(self, subg):
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID], new_g.ndata['memory'])
        self.memory.set_last_update_t(
            new_g.ndata[dgl.NID], new_g.ndata['timestamp'])

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()
        self.memory = self.memory.to(self.device)

    def store_memory(self):
        memory_checkpoint = {}
        memory_checkpoint['memory'] = copy.deepcopy(self.memory.memory)
        memory_checkpoint['last_t'] = copy.deepcopy(self.memory.last_update_t)
        return memory_checkpoint

    def restore_memory(self, memory_checkpoint):
        self.memory.memory = memory_checkpoint['memory']
        self.memory.last_update_time = memory_checkpoint['last_t']
