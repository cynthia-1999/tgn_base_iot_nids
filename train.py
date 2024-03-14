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
from utils.data_preprocess import MyTemporalDataset, TemporalBotiotDataset, TemporalToniotDataset
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

def train(model, contrast_op, dataloader, sampler, criterion, optimizer, args, device, batch_folder):
    model.train()
    total_loss = 0
    batch_cnt = 0
    train_acc = []
    last_t = time.time()
    batch_files = [f for f in os.listdir(batch_folder) if f.endswith('.bin')]
    # for _, g, blocks in dataloader:
        # dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
    for batch_file in batch_files:
        file_path = os.path.join(batch_folder, batch_file)
        gs, _ = dgl.load_graphs(file_path)
        g, block = gs[0], gs[1]
        blocks = []
        blocks.append(block)
        g = g.to(device)
        model.update_memory(g)
        blocks[0] = blocks[0].to(device)
        optimizer.zero_grad()

        '''
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
        '''
        # predict
        embeddings = model.embed(g, blocks)
        predictions, _ = model.predict(g, embeddings)
        labels = g.edata['label']
        labels = labels.type(torch.LongTensor).to(device)
        loss = criterion(predictions, labels)
        # loss += criterion(predictions, labels)
        total_loss += float(loss)*args.batch_size
        retain_graph = True if batch_cnt == 0 else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.detach_memory()
        
        train_acc.append(compute_accuracy(predictions, labels))
        print("Batch: ", batch_cnt, "Time: ", time.time()-last_t)
        last_t = time.time()
        batch_cnt += 1
    return total_loss, float(torch.tensor(train_acc).mean())

def test(model, dataloader, sampler, criterion, args, device, batch_folder, classes):
    model.eval()
    batch_size = args.batch_size
    total_loss = 0
    actuals, test_preds, test_embeddings, test_scores = [], [], [], []
    test_acc = []
    batch_cnt = 0
    batch_files = [f for f in os.listdir(batch_folder) if f.endswith('.bin')]
    # label2actual = ['']
    if classes == 2:
        label2actual = ["Normal", "Attack"]
    else:
        label2actual = np.load(batch_folder + '/actual.npy', allow_pickle=True)
    with torch.no_grad():
        # for _, g, blocks in dataloader:
        #     dgl.save_graphs(batch_folder + '/batch{}.bin'.format(batch_cnt), [g, blocks[0]])
        for batch_file in batch_files:
            file_path = os.path.join(batch_folder, batch_file)
            gs, _ = dgl.load_graphs(file_path)
            g, block = gs[0], gs[1]
            blocks = []
            blocks.append(block)
            g = g.to(device)
            model.update_memory(g)
            blocks[0] = blocks[0].to(device)
            embeddings = model.embed(g, blocks)
            predictions, edge_embs = model.predict(g, embeddings)
            labels = g.edata['label']
            labels = labels.type(torch.LongTensor).to(device)
            loss = criterion(predictions, labels)
            total_loss += float(loss)*batch_size

            test_acc.append(compute_accuracy(predictions, labels))

            test_pred = predictions.argmax(1).cpu().numpy().astype(int)
            labels_numpy = labels.cpu().numpy().astype(int)
            test_actual = [label2actual[label] for label in labels_numpy]
            test_pred = [label2actual[i] for i in test_pred]
            
            actuals.append(test_actual)
            test_preds.append(test_pred)
            test_embeddings.append(edge_embs.cpu().numpy())
            test_scores.append(predictions.cpu().numpy())
            batch_cnt += 1

    return float(torch.tensor(test_acc).mean()), actuals, test_preds, label2actual, total_loss, test_embeddings, test_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_class", action="store_true", default=False,
                        help="test for multi classes")
    parser.add_argument("--epochs", type=int, default=5,
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

    project_name = 'project_' + str(styleTime)
    # comments=f"dataset({args.dataset})_multiclas({args.multi_class})_bsize({args.batch_size})_embeddingdim({args.embedding_dim}_projdim({args.proj_dim})_"
    comments=f"dataset({args.dataset})_multiclas({args.multi_class})_SplyByClasses({args.split_by_classes})"
    # dir_checkpoint = Path(f"./cache/{project_name}_{comments}/checkpoints/")
    writer = SummaryWriter(comment=comments)
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
        classes = 5 if args.multi_class else 2
    elif args.dataset == 'ToN-IoT':
        data = TemporalToniotDataset(args.multi_class)
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
        
        # eids = torch.arange(0, num_edges)
        # train_edge_mask = (data.edata['timestamp'] <= test_split_ts) + (data.edata['timestamp'] <= 0)
        # test_edge_mask = data.edata['timestamp'] > test_split_ts

        # train_seed, test_seed = eids[train_edge_mask], eids[test_edge_mask]
        
        train_seed = np.where(data.edata['timestamp'] <= test_split_ts)[0] + np.where(data.edata['timestamp'] <= 0)[0]
        test_seed = np.where(data.edata['timestamp'] > test_split_ts)[0]
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

    model = TGN(edge_feat_dim=edge_dim,
                out_classes=classes,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_nodes,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop, device=device).to(device)
    
    contrast_op = ContrastModule(args.embedding_dim, args.proj_dim).to(device)
    # criterion = torch.nn.BCEWithLogitsLoss()
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', 
                                                classes = np.unique(data.edata['label'].numpy()), y = data.edata['label'].numpy())
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("class_weights.dtype:", class_weights.dtype)
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Implement Logging mechanism

    dataset_path = "/root/zc/tgn_base_iot_nids/datasets/" + args.dataset + "/"
    saved_folder = dataset_path + ("saved_multiclass/" if args.multi_class else "saved_binary/")
    if args.split_by_classes:
        saved_folder = saved_folder + "split_by_classes/"
    else:
        saved_folder = saved_folder + "split_by_time/"
    train_batch_folder = saved_folder + "train_batch"
    test_batch_folder = saved_folder + "test_batch"
    
    try:
        for i in range(args.epochs):
            print("epoch ", i)
            train_loss, train_acc = train(model, contrast_op, train_dataloader, sampler, 
                               criterion, optimizer, args, device, train_batch_folder)
            writer.add_scalar("train_loss/train", train_loss, i)
            test_acc, test_actuals, test_preds, labels, total_test_loss, embs, scores = test(
                model, test_dataloader, sampler, criterion, args, device, test_batch_folder, classes)
            
            # save edge embs
            embs = [inner for outer in embs for inner in outer]
            emb_num = np.array(embs)
            np.save(str(destination_folder)+f"/epoch{i}_embs.npy", emb_num, allow_pickle=True)
            
            # save edge scores
            scores = [inner for outer in scores for inner in outer]
            scores_num = np.array(scores)
            np.save(str(destination_folder)+f"/epoch{i}_scores.npy", scores_num, allow_pickle=True)

            test_actuals = ' '.join([' '.join(row) for row in test_actuals]).split()
            test_preds = ' '.join([' '.join(row) for row in test_preds]).split()

            # save labels
            if i == 0:
                test_actuals = np.array(test_actuals)
                np.save(str(destination_folder)+f"/labels.npy", test_actuals, allow_pickle=True)
            
            # save preds
            test_preds = np.array(test_preds)
            np.save(str(destination_folder)+f"/epoch{i}_preds.npy", test_preds, allow_pickle=True)

            writer.add_scalar("test_loss/test", total_test_loss, i)
            writer.add_scalar("accuracy/test", test_acc, i)

            memory_checkpoint = model.store_memory()
            log_content = []
            log_content.append("Epoch: {}; Training Loss: {} ; train_acc = {:.4f}\n".format(
                i, train_loss, train_acc))
            log_content.append("Epoch: {}; Test Loss: {} ; test_acc = {:.4f}\n".format(
                i, total_test_loss, test_acc))
            f.writelines(log_content) 
            model.reset_memory()
            print(log_content[0], log_content[1])
    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
