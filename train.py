import argparse
import traceback
import time
import copy
from pathlib import Path
import os

import numpy as np
import dgl
import torch

from tgn import TGN
from utils.data_preprocess import MyTemporalDataset, TemporalBotiotDataset, TemporalToniotDataset
from utils.dataloading import (TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.metrics import average_precision_score, roc_auc_score
from contrast import drop_feature, ContrastModule

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

DROP_FEATURE_RATE_1 = 0.3
DROP_FEATURE_RATE_2 = 0.2


now = time.time()
timeArray = time.localtime(now)
styleTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)

# set random Seed
np.random.seed(2021)
torch.manual_seed(2021)

def my_clone_graph(g, device):
    g_copy = copy.deepcopy(g)
    # g_copy.ndata[dgl.NID] = g.ndata[dgl.NID]
    g_copy.ndata[dgl.NID] = g.nodes()
    # g_copy.edata[dgl.EID] = g.edges()
    return g_copy

def train(model, contrast_op, dataloader, sampler, criterion, optimizer, args, device):
    model.train()
    total_loss = 0
    batch_cnt = 0
    last_t = time.time()
    for _, positive_pair_g, blocks in dataloader:
        print(f"Batch: {batch_cnt} start")
        positive_pair_g = positive_pair_g.to(device)
        blocks[0] = blocks[0].to(device)

        optimizer.zero_grad()
        # 对比学习
        # ToDo: 尝试不同的对比目标
        g_feature = positive_pair_g.edata['feats']
        g_feature_1 = drop_feature(g_feature, DROP_FEATURE_RATE_1)
        g_feature_2 = drop_feature(g_feature, DROP_FEATURE_RATE_2)

        positive_pair_g.edata['feats'] = g_feature_1
        z1 = model.embed(positive_pair_g, blocks)
        positive_pair_g.edata['feats'] = g_feature_2
        z2 = model.embed(positive_pair_g, blocks)

        loss = contrast_op.loss(z1, z2, batch_size=0)
 
        positive_pair_g.edata['feats'] = g_feature

        # predict
        embeddings = model.embed(positive_pair_g, blocks)
        predictions = model.predict(positive_pair_g, embeddings)
        labels = positive_pair_g.edata['label'].unsqueeze(1)
        loss += criterion(predictions, labels)
        # loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss)*args.batch_size
        retain_graph = True if batch_cnt == 0 else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.detach_memory()
        # ToDo: 更新内存的时机应该在预测之前
        if not args.not_use_memory:
            model.update_memory(positive_pair_g)
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss

def test_val(model, dataloader, sampler, criterion, args, device):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    aps, aucs = [], []
    batch_cnt = 0
    with torch.no_grad():
        # for _, positive_pair_g, negative_pair_g, blocks in dataloader:
        for _, positive_pair_g, blocks in dataloader:
            positive_pair_g = positive_pair_g.to(device) 
            blocks[0] = blocks[0].to(device)
            embeddings = model.embed(positive_pair_g, blocks)
            predictions = model.predict(positive_pair_g, embeddings)
            # pred_pos = model.embed(positive_pair_g, blocks)
            labels = positive_pair_g.edata['label'].unsqueeze(1)
            loss = criterion(predictions, labels)
            total_loss += float(loss)*batch_size
            y_pred = predictions.sigmoid().cpu()
            y_true = labels.cpu()
            if not args.not_use_memory:
                model.update_memory(positive_pair_g)
            aps.append(average_precision_score(y_true, y_pred))
            # aucs.append(roc_auc_score(y_true, y_pred))
            batch_cnt += 1
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_class", action="store_true", default=False,
                        help="test for multi classes")
    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs for training on entire dataset')
    parser.add_argument("--device_id", type=int,
                        default=3, help="gpu device id")
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--proj_dim", type=int, default=100,
                        help="project dim for Contrast")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--dataset", type=str, default="wikipedia",
                        help="dataset selection wikipedia/reddit")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")


    args = parser.parse_args()
    

    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    project_name = 'project_' + str(styleTime)
    # comments=f"dataset({args.dataset})_multiclas({args.multi_class})_bsize({args.batch_size})_embeddingdim({args.embedding_dim}_projdim({args.proj_dim})_"
    comments=f"dataset({args.dataset})_multiclas({args.multi_class})"
    # dir_checkpoint = Path(f"./cache/{project_name}_{comments}/checkpoints/")
    destination_folder = Path(f"./cache/{project_name}_{comments}/")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    log_file = os.path.join(destination_folder, os.path.basename("logging.txt"))
    f = open(log_file, 'w')
    log_content = []
    

    device_str = "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("device:", device_str)
    
    if args.dataset == 'BoT-IoT':
        data = TemporalBotiotDataset(args.multi_class)
    elif args.dataset == 'ToN-IoT':
        data = TemporalToniotDataset(args.multi_class)
    else:
        print("Warning Using Untested Dataset: "+args.dataset)
        data = MyTemporalDataset(args.dataset)

    # data = data.to(device)
    # Pre-process data, mask new node in test set from original graph
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    print("num_nodes:", num_nodes)
    print("num_edges:", num_edges)

    trainval_div = int(VALID_SPLIT*num_edges)
    print("trainval_div:", trainval_div)

    # Select new node from test set and remove them from entire graph
    test_split_ts = data.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([data.edges()[0][trainval_div:], data.edges()[
                           1][trainval_div:]]).unique().numpy()
    test_new_nodes = np.random.choice(
        test_nodes, int(0.1*len(test_nodes)), replace=False)
    print("test_new_nodes:", test_new_nodes)

    # indegree
    in_subg = dgl.in_subgraph(data, test_new_nodes)
    # outdegree
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    new_node_eid_delete = torch.cat(
        [new_node_in_eid_delete, new_node_out_eid_delete]).unique()

    graph_new_node = copy.deepcopy(data)
    # relative order preseved
    graph_new_node.remove_edges(new_node_eid_delete)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(data)
    graph_no_new_node.remove_edges(eid_delete)
    # graph_no_new_node.remove_nodes(test_new_nodes) # 不能删除节点，因为memory中的索引不能变。
    # graph_no_new_node and graph_new_node should have same set of nid

    # data = data.to(device)
    # graph_no_new_node = graph_no_new_node.to(device)
    # graph_new_node = graph_new_node.to(device)
    # Sampler Initialization
    sampler = TemporalSampler(k=args.n_neighbors)
    edge_collator = TemporalEdgeCollator

    # ToDo: remove negative edge
    # neg_sampler = dgl.dataloading.negative_sampler.Uniform(
    #     k=0)
    neg_sampler = None
    
    # Set Train, validation, test and new node test id
    #  
    # 不包含新节点的训练数据
    # train_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges())).to(device)
    train_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()))
    # 不包含新节点的验证数据
    # valid_seed = torch.arange(int(
    #     TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0)).to(device)
    valid_seed = torch.arange(int(
        TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    # 不包含新节点的测试数据
    # test_seed = torch.arange(
    #     trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges()).to(device)
    test_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    # 包含新节点的测试数据
    # test_new_node_seed = torch.arange(
    #     trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges()).to(device)
    test_new_node_seed = torch.arange(
        trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges())

    # ToDo: if need add_reverse_edges?
    # g_sampling = dgl.add_reverse_edges(graph_no_new_node, copy_edata=True)
    # new_node_g_sampling = dgl.add_reverse_edges(graph_new_node, copy_edata=True)

    g_sampling = graph_no_new_node
    new_node_g_sampling = graph_new_node
    new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
    g_sampling.ndata[dgl.NID] = g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    print("g_sampling:", g_sampling)
    print("new_node_g_sampling:", new_node_g_sampling)
    train_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              train_seed,
                                              sampler,
                                            #   device_str,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    valid_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              valid_seed,
                                              sampler,
                                            #   device_str,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    test_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                             test_seed,
                                             sampler,
                                            #  device_str,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling)

    test_new_node_dataloader = TemporalEdgeDataLoader(graph_new_node,
                                                      test_new_node_seed,
                                                      sampler,
                                                    #   device_str,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=new_node_g_sampling)

    edge_dim = data.edata['feats'].shape[1]
    num_node = data.num_nodes()

    print("edge_dim:", edge_dim)
    print("memory_dim:", args.memory_dim)
    print("temporal_dim:", args.temporal_dim)
    print("embedding_dim:", args.embedding_dim)
    print("proj_dim:", args.proj_dim)
    print("num_heads:", args.num_heads)
    print("num_nodes:", num_nodes)
    print("n_neighbors:", args.n_neighbors)
    print("memory_updater:", args.memory_updater)

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop, device=device).to(device)
    
    contrast_op = ContrastModule(args.embedding_dim, args.proj_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Implement Logging mechanism
    
    try:
        for i in range(args.epochs):
            print("epoch ", i)
            train_loss = train(model, contrast_op, train_dataloader, sampler, 
                               criterion, optimizer, args, device)
            val_ap, val_auc = test_val(
                model, valid_dataloader, sampler, criterion, args, device)
            memory_checkpoint = model.store_memory()
            test_ap, test_auc = test_val(
                model, test_dataloader, sampler, criterion, args, device)
            model.restore_memory(memory_checkpoint)
            sample_nn = sampler
            nn_test_ap, nn_test_auc = test_val(
                model, test_new_node_dataloader, sample_nn, criterion, args, device)
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                i, train_loss, val_ap, val_auc))
            log_content.append(
                "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
            log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                i, nn_test_ap, nn_test_auc))

            f.writelines(log_content)
            model.reset_memory()
            print(log_content[0], log_content[1], log_content[2])
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
