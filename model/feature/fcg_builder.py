import pickle
import json
import numpy as np
import torch
from torch.cuda import current_device
from torch.nn import Linear

from config import NODE_EMBEDDING, EDGE_MATRIX, data_path

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')
VIRTUAL_EDGE_WEIGHT = 0.1  # TODO: 移至 config.py

class FCGBuilder:
    """构建细粒度常识图(Finegrained Commonsense Graph)"""
    
    def __init__(self, hidden_dim):
        # 加载词嵌入和边统计矩阵
        with open(NODE_EMBEDDING, 'rb') as f:
            self.emb_ent, self.emb_pred = pickle.load(f)
        self.edge_matrix = np.load(EDGE_MATRIX) # 151x151x51
        self.emb_fc = Linear(self.emb_ent.size(1) * 2 + self.emb_pred.size(1), hidden_dim)
        
        # 构建节点和边
        self.build_nodes()
        self.build_edges()
        # 保存元数据
        self.save_metadata()
        
    def build_nodes(self):
        """构建三级节点,包括真实节点和虚节点"""
        # 第三级:具体三元组节点,如<person,standing on,snow>
        self.level3_nodes = []  # 存储(s,p,o)三元组
        self.level3_feats = []  # 存储节点特征
        self.level3_virtual = [] # 存储是否为虚节点
        self.level3_freq = []   # 存储节点频次
        
        # 第二级:谓词模式节点,如<person,standing on,X>和<X,standing on,snow>
        self.level2_nodes = []  # 存储(s,p)或(p,o)二元组
        self.level2_feats = []  # 存储节点特征
        
        # 第一级:实体模式节点,如<person,X,X>和<X,X,snow>
        self.level1_nodes = []  # 存储s或o
        self.level1_feats = []  # 存储节点特征
        
        # 1. 首先添加真实节点
        for s in range(151):
            for o in range(151):
                # 添加一级节点(如果不存在)
                if s not in self.level1_nodes:
                    self.level1_nodes.append(s)
                    self.level1_feats.append(torch.tensor(self.emb_ent[s]))
                if o not in self.level1_nodes:
                    self.level1_nodes.append(o)
                    self.level1_feats.append(torch.tensor(self.emb_ent[o]))
                
                for p in range(51):
                    freq = self.edge_matrix[s,o,p]
                    if freq > 0:
                        # 添加三级节点
                        node_feat = torch.cat([
                            torch.tensor(self.emb_ent[s]),
                            torch.tensor(self.emb_pred[p]), 
                            torch.tensor(self.emb_ent[o])
                        ])
                        self.level3_nodes.append((s,p,o))
                        self.level3_feats.append(node_feat)
                        self.level3_virtual.append(False)
                        self.level3_freq.append(freq)
                        
                        # 添加二级节点
                        if (s,p) not in self.level2_nodes:
                            self.level2_nodes.append((s,p))
                        if (p,o) not in self.level2_nodes:
                            self.level2_nodes.append((p,o))

        # 2. 构建虚节点
        for s in range(151):
            for o in range(151):
                if not any(self.edge_matrix[s,o,:] > 0):
                    # 寻找桥接谓词
                    bridge_preds = set()
                    for x in range(151):
                        for p in range(51):
                            if (self.edge_matrix[s,x,p] > 0 and 
                                self.edge_matrix[x,o,p] > 0):
                                bridge_preds.add(p)
                    
                    # 为每个桥接谓词创建虚节点            
                    for p in bridge_preds:
                        node_feat = torch.cat([
                            torch.tensor(self.emb_ent[s]),
                            torch.tensor(self.emb_pred[p]),
                            torch.tensor(self.emb_ent[o])
                        ])
                        self.level3_nodes.append((s,p,o))
                        self.level3_feats.append(node_feat)
                        self.level3_virtual.append(True)
                        self.level3_freq.append(0)
                        
                        # 虚节点也要添加对应的二级节点
                        if (s,p) not in self.level2_nodes:
                            self.level2_nodes.append((s,p))
                        if (p,o) not in self.level2_nodes:
                            self.level2_nodes.append((p,o))
                            
        # 转换为tensor
        self.level3_feats = torch.stack(self.level3_feats)
        self.level3_virtual = torch.tensor(self.level3_virtual, dtype=torch.bool)
        self.level3_freq = torch.tensor(self.level3_freq)
        self.level1_feats = torch.stack(self.level1_feats)
        
        # 计算二级节点特征(子节点平均)
        self.level2_feats = []
        for s,p in self.level2_nodes:
            child_feats = []
            for i, (s2,p2,o2) in enumerate(self.level3_nodes):
                if s==s2 and p==p2 and not self.level3_virtual[i]:
                    child_feats.append(self.level3_feats[i])
            if child_feats:
                self.level2_feats.append(torch.stack(child_feats).mean(0))
            else:
                virtual_feats = []
                for i, (s2,p2,o2) in enumerate(self.level3_nodes):
                    if s==s2 and p==p2:
                        virtual_feats.append(self.level3_feats[i])
                self.level2_feats.append(torch.stack(virtual_feats).mean(0))
        self.level2_feats = torch.stack(self.level2_feats)

    def build_edges(self):
        """构建层级之间的边连接"""
        # 一级到二级的边
        self.edges_l1_l2 = torch.zeros(len(self.level1_nodes), len(self.level2_nodes))
        for i,s1 in enumerate(self.level1_nodes):
            for j,(s2,p2) in enumerate(self.level2_nodes):
                if s1 == s2:  # 主语模式
                    self.edges_l1_l2[i,j] = 1
                
        # 二级到三级的边
        self.edges_l2_l3 = torch.zeros(len(self.level2_nodes), len(self.level3_nodes))
        for i,(s1,p1) in enumerate(self.level2_nodes):
            for j,(s2,p2,o2) in enumerate(self.level3_nodes):
                if s1==s2 and p1==p2:
                    if self.level3_virtual[j]:
                        freq = VIRTUAL_EDGE_WEIGHT
                    else:
                        freq = self.edge_matrix[s2,o2,p2]
                    self.edges_l2_l3[i,j] = freq
                    
        # 转移到GPU
        self.edges_l1_l2 = self.edges_l1_l2.to(CUDA_DEVICE)
        self.edges_l2_l3 = self.edges_l2_l3.to(CUDA_DEVICE)
        
    def save_metadata(self):
        """保存FCG图的元数据到json文件"""
        metadata = {
            'level1_nodes': [
                {
                    'index': i,
                    'entity': int(node),
                    'pattern': f'<{node},X,X>' if i < len(self.level1_nodes)//2 
                             else f'<X,X,{node}>'
                } for i, node in enumerate(self.level1_nodes)
            ],
            'level2_nodes': [
                {
                    'index': i,
                    'subject': int(s),
                    'predicate': int(p),
                    'pattern': f'<{s},{p},X>'
                } for i, (s,p) in enumerate(self.level2_nodes)
            ],
            'level3_nodes': [
                {
                    'index': i,
                    'subject': int(s),
                    'predicate': int(p),
                    'object': int(o),
                    'frequency': int(self.level3_freq[i]),
                    'is_virtual': bool(self.level3_virtual[i])
                } for i, (s,p,o) in enumerate(self.level3_nodes)
            ]
        }
        
        with open(data_path('fcg_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2) 