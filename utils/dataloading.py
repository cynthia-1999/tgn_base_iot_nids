import torch
import dgl

from dgl._dataloading.dataloader import EdgeCollator
from dgl._dataloading import BlockSampler
from dgl._dataloading.pytorch import _pop_subgraph_storage, _pop_storages, EdgeDataLoader
from dgl.base import DGLError

from functools import partial
import copy
import dgl.function as fn
import numpy as np

def print_all_nodes(g):
    all_nodes = g.nodes()
    for node in all_nodes:
        print(node)

def print_all_edges(g):
    all_edges = g.edges()
    for i, edge in enumerate(zip(all_edges[0], all_edges[1])):
    # for src, dst in all_edges:
        print(f"Edge from Node {edge[0]} to Node {edge[1]}")

def print_selected_edges(g, selected_indices):
    all_edges = g.edges()
    srcs = all_edges[0]
    dsts = all_edges[1]
    # edges = [(src, dst) for src, dst in all_edges] 
    for index in selected_indices:
        print(f"Edge[{index}]: ({srcs[index]}, {dsts[index]})")

def count_edges(g, seed_nodes):
    count = 0
    all_edges = g.edges()
    for i, edge in enumerate(zip(all_edges[0], all_edges[1])):
    # for src, dst in all_edges:
        if edge[0] == seed_nodes[0] or edge[0] == seed_nodes[1] or edge[0] == seed_nodes[1] or edge[1] == seed_nodes[1]:
            count = count+1
    return count

        

def _prepare_tensor(g, data, name, is_distributed):
    return torch.tensor(data) if is_distributed else dgl.utils.prepare_tensor(g, data, name)

class TemporalSampler(BlockSampler):
    """ Temporal Sampler builds computational and temporal dependency of node representations via
    temporal neighbors selection and screening.

    The sampler expects input node to have same time stamps, in the case of TGN, it should be
    either positive [src,dst] pair or negative samples. It will first take in-subgraph of seed
    nodes and then screening out edges which happen after that timestamp. Finally it will sample
    a fixed number of neighbor edges using random or topk sampling.

    Parameters
    ----------
    sampler_type : str
        sampler indication string of the final sampler.

        If 'topk' then sample topk most recent nodes

        If 'uniform' then uniform randomly sample k nodes

    k : int
        maximum number of neighors to sampler

        default 10 neighbors as paper stated

    Examples
    ----------
    Please refers to examples/pytorch/tgn/train.py

    """

    def __init__(self, sampler_type='topk', k=10):
        super(TemporalSampler, self).__init__(1, False)
        if sampler_type == 'topk':
            self.sampler = partial(
                dgl.sampling.select_topk, k=k, weight='timestamp')
        elif sampler_type == 'uniform':
            self.sampler = partial(dgl.sampling.sample_neighbors, fanout=k)
        else:
            raise DGLError(
                "Sampler string invalid please use \'topk\' or \'uniform\'")

    # ToDo: 修改采样策略，使得一个batch的graph不能包含太多的节点，需要确定这个阈值
    def sampler_frontier(self,
                         block_id,
                         g,
                         seed_nodes,
                         current_edge_index,
                         timestamp):
        # ToDo：src和dst要单独进行采样

        selected_edges = torch.empty(0, dtype=torch.int64)
        for seed_node in seed_nodes:
            full_neighbor_subgraph = dgl.in_subgraph(g, seed_node)
            # full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
            #                                     seed_nodes, seed_nodes)
            current_selected_edges = full_neighbor_subgraph.edata[dgl.EID]

            temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (
                full_neighbor_subgraph.edata['timestamp'] <= 0)
            true_indices = np.where(temporal_edge_mask)[0]

            if len(true_indices) > 100:
                # print("sampled edges > 100")
                selected_indices = np.random.choice(true_indices, 100, replace=False)
                temporal_edge_mask = np.zeros_like(temporal_edge_mask, dtype=bool)
                temporal_edge_mask[selected_indices] = True
                temporal_edge_mask = torch.from_numpy(temporal_edge_mask)
            
            current_selected_edges = current_selected_edges[temporal_edge_mask]
            selected_edges = torch.cat((selected_edges, current_selected_edges), dim=0)

        selected_edges = torch.cat((selected_edges, torch.tensor([current_edge_index])), dim=0)
        # selected_edges = torch.tensor([current_edge_index])

        # temporal_subgraph = dgl.edge_subgraph(full_neighbor_subgraph, temporal_edge_mask)
        temporal_subgraph = dgl.edge_subgraph(g, selected_edges)
        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]

        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        # print("temporal_subgraph:", temporal_subgraph)
        final_subgraph = self.sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        # print("final_subgraph:", final_subgraph)
        return final_subgraph
        # Temporal Subgraph
        
    
    # 生成用于图神经网络训练的数据块
    def sample_blocks(self,
                      g,
                      seed_nodes,
                      current_edge_index,
                      timestamp):
        blocks = []
        frontier = self.sampler_frontier(0, g, seed_nodes, current_edge_index, timestamp)
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        if self.return_eids:
            self.assign_block_eids(block, frontier)
        blocks.append(block)
        return blocks


class TemporalEdgeCollator(EdgeCollator):
    """ Temporal Edge collator merge the edges specified by eid: items

    Since we cannot keep duplicated nodes on a graph we need to iterate though
    the incoming edges and expand the duplicated node and form a batched block
    graph capture the temporal and computational dependency.

    Parameters
    ----------

    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.

    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.

    graph_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.

    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.

    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, which excludes nothing.

        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.

        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.

        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.

    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.

    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.

    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:

        * The original (heterogeneous) graph.

        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.

        It should return

        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.

        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.

    example
    ----------
    Please refers to examples/pytorch/tgn/train.py

    """
    def __init__(self, g, eids, graph_sampler, device='cpu', g_sampling=None, exclude=None,
                reverse_eids=None, reverse_etypes=None, negative_sampler=None):
        super(TemporalEdgeCollator, self).__init__(g, eids, graph_sampler,
                                                         g_sampling, exclude, reverse_eids, reverse_etypes, negative_sampler)
        self.device = torch.device(device)
    
    def _collate(self, items):
        items = _prepare_tensor(self.g_sampling, items, 'items', False)
        # Here node id will not change
        pair_graph = self.g.edge_subgraph(items, relabel_nodes=False)
        induced_edges = pair_graph.edata[dgl.EID]
        pair_graph = dgl.transforms.compact_graphs(pair_graph)

        # Need to remap id
        pair_graph.ndata[dgl.NID] = self.g.nodes()[pair_graph.ndata[dgl.NID]]
        pair_graph.edata[dgl.EID] = induced_edges

        batch_graphs = []
        nodes_id = []

        for i, edge in enumerate(zip(self.g.edges()[0][items], self.g.edges()[1][items])):
            ts = pair_graph.edata['timestamp'][i]
            subg = self.graph_sampler.sample_blocks(self.g_sampling,
                                                    list(edge),
                                                    items[i].item(),
                                                    timestamp=ts)[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            nodes_id.append(subg.srcdata[dgl.NID])
            batch_graphs.append(subg)
        blocks = [dgl.batch(batch_graphs)]
        print("blocks:", blocks)
        # input_nodes = torch.cat(nodes_id).to(self.device)
        input_nodes = torch.cat(nodes_id)
        return input_nodes, pair_graph, blocks

    # 将边ID列表转换为一个数据块，并在其中执行一些操作，以准备好用于图神经网络的训练或评估。这些操作包括移除与输入图和采样图相关的子图存储，可能是为了减少内存占用和优化性能。
    def collator(self, items):
        """
        The interface of collator, input items is edge id of the attached graph
        """
        result = super().collate(items)
        # Copy the feature from parent graph
        _pop_subgraph_storage(result[1], self.g)
        _pop_subgraph_storage(result[2], self.g)
        _pop_storages(result[-1], self.g_sampling)
        return result

# 用于处理带有时间信息的边数据的数据加载器，他生成时间顺序排列的数据块，用于时间嵌入模型的训练
class TemporalEdgeDataLoader(EdgeDataLoader):
    """ TemporalEdgeDataLoader is an iteratable object to generate blocks for temporal embedding
    as well as pos and neg pair graph for memory update.

    The batch generated will follow temporal order

    Parameters
    ----------
    g : dgl.Heterograph
        graph for batching the temporal edge id as well as generate negative subgraph

    eids : torch.tensor() or numpy array
        eids range which to be batched, it is useful to split training validation test dataset

    graph_sampler : dgl.dataloading.BlockSampler
        temporal neighbor sampler which sample temporal and computationally depend blocks for computation

    device : str
        'cpu' means load dataset on cpu
        'cuda' means load dataset on gpu

    collator : dgl.dataloading.EdgeCollator
        Merge input eid from pytorch dataloader to graph

    Example
    ----------
    Please refers to examples/pytorch/tgn/train.py

    """

    def __init__(self, g, eids, graph_sampler, device='cpu', collator=TemporalEdgeCollator, **kwargs):
        super().__init__(g, eids, graph_sampler, device, **kwargs)
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        self.collator = collator(g, eids, graph_sampler, device, **collator_kwargs)

        assert not isinstance(g, dgl.distributed.DistGraph), \
            'EdgeDataLoader does not support DistGraph for now. ' \
            + 'Please use DistDataLoader directly.'
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)
        self.device = device

        # Precompute the CSR and CSC representations so each subprocess does not
        # duplicate.
        if dataloader_kwargs.get('num_workers', 0) > 0:
            g.create_formats_()

    def __iter__(self):
        return iter(self.dataloader)
