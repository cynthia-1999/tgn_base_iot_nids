import argparse
import traceback
import time
import copy
from pathlib import Path
import os

import numpy as np
import dgl
import torch
from collections import Counter

from tgn import TGN
from utils.data_preprocess import MyTemporalDataset, TemporalBotiotDataset, TemporalToniotDataset, TemporalToniotAllDataset
from utils.dataloading import (TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder

from contrast import drop_feature, ContrastModule
from torch.utils.tensorboard import SummaryWriter

TRAIN_SPLIT = 0.7

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

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def make_train_batch(model, contrast_op, dataloader, sampler, criterion, optimizer, args, device, batch_folder):
    total_loss = 0
    batch_cnt = 0
    train_acc = []
    last_t = time.time()
    for _, g, blocks in dataloader:
        dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss, float(torch.tensor(train_acc).mean())

def make_test_batch(model, dataloader, sampler, criterion, args, device, batch_folder, classes):
    total_loss = 0
    actuals, test_preds, test_embeddings, test_scores = [], [], [], []
    test_acc = []
    batch_cnt = 0
    label2actual = []
    with torch.no_grad():
        for _, g, blocks in dataloader:
            dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
            batch_cnt += 1
    return float(torch.tensor(test_acc).mean()), actuals, test_preds, label2actual, total_loss, test_embeddings, test_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_class", action="store_true", default=False,
                        help="test for multi classes")
    parser.add_argument("--epochs", type=int, default=1,
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
    parser.add_argument("--dataset", type=str, default="BoT-IoT",
                        help="dataset selection BoT-IoT/ToN-IoT")
    parser.add_argument("--split_by_classes", action="store_true", default=False,
                        help="Split the training and test sets by classes")
    parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")


    args = parser.parse_args()
    

    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode" 

    device_str = "cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("device:", device_str)

    if args.dataset == 'BoT-IoT':
        data = TemporalBotiotDataset(args.multi_class)
        classes = 5 if args.multi_class else 2
    elif args.dataset == 'ToN-IoT':
        data = TemporalToniotDataset(args.multi_class)
        classes = 10 if args.multi_class else 2
    elif args.dataset == 'ToN-IoT-All':
        data = TemporalToniotAllDataset(args.multi_class)
        classes = 10 if args.multi_class else 2
    else:
        print("Warning Using Untested Dataset: "+args.dataset)
        data = MyTemporalDataset(args.dataset)

    # Sampler Initialization
    sampler = TemporalSampler(k=args.n_neighbors)
    edge_collator = TemporalEdgeCollator

    neg_sampler = None

    print("data:", data)
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()
    if not args.split_by_classes:
        traintest_div = int(TRAIN_SPLIT*num_edges)
        test_split_ts = data.edata['timestamp'][traintest_div]
        
        eids = torch.arange(0, num_edges)
        train_edge_mask = (data.edata['timestamp'] <= test_split_ts) + (data.edata['timestamp'] <= 0)
        test_edge_mask = data.edata['timestamp'] > test_split_ts

        train_seed, test_seed = eids[train_edge_mask], eids[test_edge_mask]
        
        # train_seed = np.where(data.edata['timestamp'] <= test_split_ts)[0] + np.where(data.edata['timestamp'] <= 0)[0]
        # test_seed = np.where(data.edata['timestamp'] > test_split_ts)[0]
    else:
        # 从data的边中按照labels均匀采样出70%的边
        labels = data.edata['label'].numpy()
        label_counter = Counter(labels)
        print("label_counter:", label_counter)

        train_seed = []
        for label, ratio in label_counter.items():
            label_edges = np.where(data.edata['label'] == label)[0]
            print("label_edges.shape", label_edges.shape)
            num_label_edges = int(TRAIN_SPLIT * len(label_edges))
            print("num_label_edges:", num_label_edges)
            train_seed.extend(np.random.choice(label_edges, num_label_edges, replace=False))

        train_seed = np.sort(train_seed)

        test_seed = np.array(list((set(np.arange(0, num_edges)) - set(train_seed))))

    print("train_seed:", train_seed)
    print("len(train_seed):", len(train_seed))
    print("test_seed:", test_seed)
    print("len(test_seed):", len(test_seed))

    g_sampling = data   
    g_sampling.ndata[dgl.NID] = g_sampling.nodes()


    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.
    train_dataloader = TemporalEdgeDataLoader(data,
                                              train_seed,
                                              sampler,
                                            #   device_str,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=16,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    test_dataloader = TemporalEdgeDataLoader(data,
                                             test_seed,
                                             sampler,
                                            #  device_str,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=16,
                                             collator=edge_collator,
                                             g_sampling=g_sampling)


    edge_dim = data.edata['feats'].shape[1]

    print("edge_dim:", edge_dim)
    print("memory_dim:", args.memory_dim)
    print("temporal_dim:", args.temporal_dim)
    print("embedding_dim:", args.embedding_dim)
    print("proj_dim:", args.proj_dim)
    print("num_heads:", args.num_heads)
    print("num_nodes:", num_nodes)
    print("num_edges:", num_edges)
    print("n_neighbors:", args.n_neighbors)
    print("memory_updater:", args.memory_updater)

    model = 0
    
    contrast_op = 0
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = 0
    optimizer = 0
    # Implement Logging mechanism

    dataset_path = "/root/zc/tgn_base_iot_nids/datasets/" + args.dataset + "/"
    saved_folder = dataset_path + ("saved_multiclass/" if args.multi_class else "saved_binary/")
    if args.split_by_classes:
        saved_folder = saved_folder + "split_by_classes/"
    else:
        saved_folder = saved_folder + "split_by_time/"
    train_batch_folder = saved_folder + "train_batch_new"
    test_batch_folder = saved_folder + "test_batch_new"
    
    try:
        for i in range(args.epochs):
            print("epoch ", i)
            train_loss, train_acc = make_train_batch(model, contrast_op, train_dataloader, sampler, 
                               criterion, optimizer, args, device, train_batch_folder)
            
            test_acc, test_actuals, test_preds, labels, total_test_loss, embs, scores = make_test_batch(
                model, test_dataloader, sampler, criterion, args, device, test_batch_folder, classes)     
            # '''
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
    print("========Training is Done========")
