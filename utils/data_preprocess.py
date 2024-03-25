import os
import ssl
from six.moves import urllib

import pandas as pd
import numpy as np
import networkx as nx

import torch
import dgl
from dgl import from_networkx
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from utils.edit_table import move_column_to_position_in_data

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def drop_edge(edge_index, drop_prob):
    drop_mask = torch.empty(
        (edge_index.size(0)),
        dtype=torch.float32,
        device=edge_index.device).uniform_(0, 1) < drop_prob
    remove_edge_index = torch.arange(edge_index.size(0), dtype=torch.int64, device=edge_index.device)
    remove_edge_index = remove_edge_index[drop_mask]
    return remove_edge_index

# Re index nodes for DGL convience
def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite: # 如果数据集是二部图（bipartite），则用户和项会被分别重索引。
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

def my_preprocess(data_path, data_name, multi_class):
    data = pd.read_csv(data_path)
    data.drop(columns=['pkSeqID', 'flgs', 'seq', 'subcategory'],inplace=True)

    if multi_class:
        data.drop(columns=['attack'],inplace=True)
        data.rename(columns={"category": "label"},inplace = True)
        print(data.label.value_counts())
        le = LabelEncoder()
        le.fit_transform(data.label.values)
        data['label'] = le.transform(data['label'])
        # label2actual = le.inverse_transform([0, 1, 2, 3, 4])
        # np.save('/root/zc/tgn_base_iot_nids/datasets/BoT-IoT/saved_multiclass/test_batch/actual.npy', label2actual, allow_pickle=True)
    else:
        print("test")
        data.drop(columns=['category'],inplace=True)
        data.rename(columns={"attack": "label"},inplace = True)
    data['label'] = data.label.apply(int)
    
    data['saddr'] = data.saddr.apply(str)
    data['sport'] = data.sport.apply(str)
    data['daddr'] = data.daddr.apply(str)
    data['dport'] = data.dport.apply(str)

    
    data['saddr'] = data['saddr'] + ':' + data['sport']
    data['daddr'] = data['daddr'] + ':' + data['dport']
    
    data.drop(columns=['sport','dport'],inplace=True)
    data = pd.get_dummies(data, columns = ['proto', 'state'])

    data = data.reset_index()
    data.replace([np.inf, -np.inf], np.nan,inplace = True)
    data.fillna(0,inplace = True)
        
    print("data.columns:", data.columns)
    data = move_column_to_position_in_data(data, 'saddr', 1)
    data = move_column_to_position_in_data(data, 'daddr', 2)
    data = move_column_to_position_in_data(data, 'stime', 3)
    print("data.columns:", data.columns)

    scaler = StandardScaler()
    cols_to_norm = list(set(list(data.iloc[:, 4:].columns )) - set(list(['label'])))
    data[cols_to_norm] = scaler.fit_transform(data[cols_to_norm])
    
    data = move_column_to_position_in_data(data, 'label', 4)
    data.drop(columns=['index'],inplace=True)
    # print("data.label.value_counts:", data.label.value_counts())

    print("data.columns:", data.columns)
    # sort by ts
    data = data.sort_values(by='stime')
    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    for idx, row in data.iterrows():
        u = row[0]
        i = row[1]
        ts = float(row[2])
        label = float(row[3])
        feat = np.array([float(x) for x in row[4:]])
        
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        
        feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)      

def ton_preprocess(data_path, data_name, multi_class, used_all_data):
    data = pd.read_csv(data_path)
    data = data.sort_values(by='ts')
    # data = data.head(int(data.shape[0] * 0.1))
    print(data.type.value_counts())
    # if not used_all_data:
    #     mitm = data[data['type'] == 'mitm']
    #     ransomware = data[data['type'] == 'ransomware']
    #     df = data.drop(data[(data.type == 'mitm') | (data.type == 'ransomware')].index)
    #     df = df.sample(frac=0.1,random_state = 42)
    #     data = pd.concat([df,mitm,ransomware], axis=0, ignore_index=True)
    #     print(data.type.value_counts())

    scanning = data[data['type'] == 'scanning']
    ddos = data[data['type'] == 'ddos']
    dos = data[data['type'] == 'dos']
    xss = data[data['type'] == 'xss']
    password = data[data['type'] == 'password']

    scanning = scanning.head(int(scanning.shape[0] * 0.1))
    ddos = ddos.head(int(ddos.shape[0] * 0.1))
    dos = dos.head(int(dos.shape[0] * 0.1))
    xss = xss.head(int(xss.shape[0] * 0.1))
    password = password.head(int(password.shape[0] * 0.1))

    df = data.drop(data[(data.type == 'scanning') | (data.type == 'ddos') | (data.type == 'dos') | (data.type == 'xss') | (data.type == 'password')].index)
    data = pd.concat([df, scanning, ddos, dos, xss, password], axis=0, ignore_index=True)
    print(data.type.value_counts())
    data = data.sort_values(by='ts')

    data['src_ip'] = data.src_ip.apply(str)
    data['src_port'] = data.src_port.apply(str)
    data['dst_ip'] = data.dst_ip.apply(str)
    data['dst_port'] = data.dst_port.apply(str)

    data['src_ip'] = data['src_ip'] + ':' + data['src_port']
    data['dst_ip'] = data['dst_ip'] + ':' + data['dst_port']

    print(data.type.value_counts())
    print(data['http_trans_depth'].unique())

    data.drop(columns=['src_port','dst_port','http_uri', 'http_referrer', 'weird_name','weird_addl','dns_query','ssl_subject','ssl_issuer'],inplace=True)
    
    if multi_class:
        data.drop(columns=['label'],inplace=True)
        data.rename(columns={"type": "label"},inplace = True)
        print(data.label.value_counts())
        le = LabelEncoder()
        le.fit_transform(data.label.values)
        data['label'] = le.transform(data['label'])
        # label2actual = le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # np.save('/root/zc/tgn_base_iot_nids/datasets/ToN-IoT/saved_multiclass/test_batch/actual.npy', label2actual, allow_pickle=True)
        # print("label2actual:", label2actual)
    else:
        data.drop(columns=['type'],inplace=True)
    data['label'] = data.label.apply(int)

    data = move_column_to_position_in_data(data, 'ts', 2)
    data = move_column_to_position_in_data(data, 'label', 3)

    label = data.label
    scaler = StandardScaler()
    # 'proto','service','conn_state' 进行独热编码
    data = pd.get_dummies(data, columns = ['proto'])

    # 下述特征进行标签编码
    # 'dns_AA','dns_RD','dns_RA','dns_rejected', 'ssl_resumed','ssl_established','weird_notice', 
    # 'http_trans_depth', 'http_user_agent','dns_rcode',
    # 'service','conn_state','dns_qclass','dns_qtype','ssl_version','ssl_cipher','http_method','http_version','http_status_code','http_orig_mime_types','http_resp_mime_types'
    encoder = ce.TargetEncoder(cols=['dns_AA','dns_RD','dns_RA','dns_rejected', 'ssl_resumed','ssl_established','weird_notice', 'http_trans_depth', 'http_user_agent','dns_rcode', 'dns_qclass','dns_qtype','ssl_version','ssl_cipher','http_method','http_version','http_status_code','http_orig_mime_types','http_resp_mime_types'])
    encoder.fit(data, label)
    data = encoder.transform(data)

    cols_to_norm = list(set(list(data.iloc[:, 4:].columns )) - set(list(['label'])))

    tmp_data = data
    for col in cols_to_norm:    
        tmp_data[col] = pd.to_numeric(tmp_data[col], errors='coerce')
        nan_indices = tmp_data[tmp_data[col].isnull()].index
        data.loc[nan_indices, col] = 0

    data[cols_to_norm] = scaler.fit_transform(data[cols_to_norm])
    print("data.columns:", data.columns)
    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    for idx, row in data.iterrows():
        u = row[0]
        i = row[1]
        ts = float(row[2])
        label = float(row[3])
        feat = np.array([float(x) for x in row[4:]])
        
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        
        feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)      

def my_run(folder, dataset, file_name, multi_class, bipartite=True):
    PATH = folder + '/{}.csv'.format(dataset)
    OUT_DF = folder + '/ml_{}.csv'.format(file_name)
    OUT_FEAT = folder + '/ml_{}.npy'.format(file_name)

    if dataset == "BoT-IoT":
        df, feat = my_preprocess(PATH, dataset, multi_class)
    elif dataset == "ToN-IoT":
        df, feat = ton_preprocess(PATH, dataset, multi_class, False)
    elif dataset == "ToN-IoT-All":
        df, feat = ton_preprocess(PATH, dataset, multi_class, True)
    # new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)

# If you have new dataset follow by same format in Jodie,
# you can directly use name to retrieve dataset

def MyTemporalDataset(dataset, multi_class):
    datapath = './datasets/' + dataset
    file_name = dataset
    if multi_class:
        file_name = file_name + "_multi_class"
    if not os.path.exists(datapath + '/{}.bin'.format(file_name)):
        print("Start Process Data ...")
        my_run(datapath, dataset, file_name, multi_class)
        raw_connection = pd.read_csv(datapath + '/ml_{}.csv'.format(file_name))
        raw_feature = np.load(datapath + '/ml_{}.npy'.format(file_name))
        
        g = nx.from_pandas_edgelist(raw_connection, "u", "i", create_using=nx.MultiDiGraph())
        g = from_networkx(g)
        g.edata['timestamp'] = torch.from_numpy(
            raw_connection['ts'].to_numpy())
        g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
        sorted_indices = torch.argsort(g.edata['timestamp'])
        g = g.edge_subgraph(sorted_indices)
        dgl.save_graphs(datapath + '/{}.bin'.format(file_name), [g])
    else:
        print("Data is exist directly loaded.")
        gs, _ = dgl.load_graphs(datapath + '/{}.bin'.format(file_name))
        g = gs[0]
    return g  
                
        
def TemporalBotiotDataset(multi_class):
    return MyTemporalDataset('BoT-IoT', multi_class)

def TemporalToniotDataset(multi_class):
    return MyTemporalDataset('ToN-IoT', multi_class)

def TemporalToniotAllDataset(multi_class):
    return MyTemporalDataset('ToN-IoT-All', multi_class)