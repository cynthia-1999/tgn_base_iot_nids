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

def make_train_batch(dataloader, batch_folder):
    batch_cnt = 0
    last_t = time.time()
    for _, g, blocks in dataloader:
        dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1

def make_test_batch(dataloader, batch_folder):
    batch_cnt = 0
    with torch.no_grad():
        for _, g, blocks in dataloader:
            dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
            batch_cnt += 1

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

    
    # Implement Logging mechanism

    dataset_path = "/root/zc/tgn_base_iot_nids/datasets/" + args.dataset + "/"
    saved_folder = dataset_path + ("saved_multiclass/" if args.multi_class else "saved_binary/")
    if args.split_by_classes:
        saved_folder = saved_folder + "split_by_classes/"
    else:
        saved_folder = saved_folder + "split_by_time/"
    train_batch_folder = saved_folder + "train_batch_new"
    test_batch_folder = saved_folder + "test_batch_new"

    if not args.multi_class:
        label2actual = ["Normal", "Attack"]
    else:
        label2actual = np.load(test_batch_folder + '/actual.npy', allow_pickle=True)

    # save raw data and
    labels = data.edata['label'].numpy().astype(int)
    feats = data.edata['feats'].numpy()
    train_edge_feats = [feats[n] for n in train_seed]
    test_edge_feats = [feats[n] for n in test_seed]

    train_edge_feats=np.array(train_edge_feats)
    test_edge_feats=np.array(test_edge_feats)
    
    train_edge_labels = [labels[n] for n in train_seed]
    test_edge_labels = [labels[n] for n in test_seed]
    
    
    train_edge_actuals = [label2actual[label] for label in train_edge_labels]
    test_edge_actuals = [label2actual[label] for label in test_edge_labels]

    train_edge_actuals = np.array(train_edge_actuals)
    test_edge_actuals = np.array(test_edge_actuals)

    np.save(str(train_batch_folder)+f"/train_raw_data.npy", train_edge_feats, allow_pickle=True)
    np.save(str(test_batch_folder)+f"/test_raw_data.npy", test_edge_feats, allow_pickle=True)
    np.save(str(train_batch_folder)+f"/train_edge_actuals.npy", train_edge_actuals, allow_pickle=True)
    np.save(str(test_batch_folder)+f"/test_edge_actuals.npy", test_edge_actuals, allow_pickle=True)


    try:
        make_train_batch(train_dataloader, train_batch_folder)
        make_test_batch(test_dataloader, classes)     
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
    print("========Training is Done========")
