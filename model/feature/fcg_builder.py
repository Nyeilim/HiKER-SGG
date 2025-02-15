import pickle
import numpy as np
import torch
from torch.nn import Linear
import pickle
import json
from config import data_path
from model.dataloaders.visual_genome import load_info

from config import NODE_EMBEDDING, EDGE_MATRIX, VG_SGG_DICT_FN


class FCG_Node:
    """FCG图中的节点类，用于统一管理节点信息"""
    def __init__(self, sub=None, pred=None, obj=None, level=None, freq=None, feat=None):
        self.freq = freq           # 词频，样本数量
        self.sub = sub             # 主语
        self.pred = pred           # 谓词
        self.obj = obj             # 宾语
        self.level = level         # 层级(1,2,3)
        self.feat = feat           # 节点特征
        
        self.is_virtual = None
        if self.level == 3 and self.freq == 0:
            self.is_virtual = True
            
        self.is_sub_pattern = None
        if self.level == 1 and (self.sub is not None and self.pred is None and self.obj is None):
            self.is_sub_pattern = True


class FCGBuilder:
    """构建细粒度常识图(Finegrained Commonsense Graph)"""
    
    def __init__(self, hidden_dim):
        # 加载词嵌入和边统计矩阵
        with open(NODE_EMBEDDING, 'rb') as f:
            self.emb_ent, self.emb_pred = pickle.load(f)
        self.edge_matrix = np.load(EDGE_MATRIX) # 151x151x51
        self.emb_fc = Linear(self.emb_ent.size(1) * 2 + self.emb_pred.size(1), hidden_dim)
        
        # 按层级存储节点
        self.l1_nodes = []  # 第一级节点
        self.l2_nodes = []  # 第二级节点
        self.l3_nodes = []  # 第三级节点(包括真实节点和虚节点)
        
        # 存储各层级节点的子节点
        self.level2_sp_subnodes = {}  # (s,p) -> [node1, node2, ...]
        self.level2_po_subnodes = {}  # (p,o) -> [node1, node2, ...]
        self.level1_s_subnodes = {}   # s -> [node1, node2, ...]
        self.level1_o_subnodes = {}   # o -> [node1, node2, ...]
        
        # 构建节点和边
        self.build_nodes()
        self.build_edges()
        
    def build_nodes(self):
        """构建FCG的所有节点"""
        # 1. 构建第三级真实节点
        level3_real_nodes = set()  # 用于记录真实的三级节点
        for s in range(151):
            for o in range(151):
                for p in range(51):
                    freq = self.edge_matrix[s,o,p]
                    if freq > 0:
                        # 计算节点特征
                        feat = self.emb_fc(torch.cat([
                            torch.tensor(self.emb_ent[s]),
                            torch.tensor(self.emb_pred[p]),
                            torch.tensor(self.emb_ent[o])
                        ]))
                        
                        node = FCG_Node(sub=s, pred=p, obj=o, level=3, freq=freq, feat=feat)
                        self.l3_nodes.append(node)
                        level3_real_nodes.add((s,p,o))
                        
        # 2. 构建第二级节点
        # 先收集每个二级节点的所有子节点
        for node in self.l3_nodes:
            # 收集(s,p)模式的子节点
            sp_key = (node.sub, node.pred)
            if sp_key not in self.level2_sp_subnodes:
                self.level2_sp_subnodes[sp_key] = []
            self.level2_sp_subnodes[sp_key].append(node)
            
            # 收集(p,o)模式的子节点
            po_key = (node.pred, node.obj)
            if po_key not in self.level2_po_subnodes:
                self.level2_po_subnodes[po_key] = []
            self.level2_po_subnodes[po_key].append(node)
            
        # 根据子节点创建二级节点
        level2_sp_nodes = {}  # (s,p) -> node
        level2_po_nodes = {}  # (p,o) -> node
        
        # 创建(s,p)模式的二级节点
        for sp_key, subnodes in self.level2_sp_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            sp_node = FCG_Node(sub=sp_key[0], pred=sp_key[1], level=2,
                             freq=total_freq, feat=avg_feat)
            level2_sp_nodes[sp_key] = sp_node
            
        # 创建(p,o)模式的二级节点
        for po_key, subnodes in self.level2_po_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            po_node = FCG_Node(pred=po_key[0], obj=po_key[1], level=2,
                             freq=total_freq, feat=avg_feat)
            level2_po_nodes[po_key] = po_node

        self.l2_nodes.extend(list(level2_sp_nodes.values()) + list(level2_po_nodes.values()))
            
        # 3. 构建第一级节点
        # 先收集每个一级节点的所有子节点
        for node in level2_sp_nodes.values():
            # 收集主语模式的子节点
            s = node.sub
            if s not in self.level1_s_subnodes:
                self.level1_s_subnodes[s] = []
            self.level1_s_subnodes[s].append(node)
            
        for node in level2_po_nodes.values():
            # 收集宾语模式的子节点
            o = node.obj
            if o not in self.level1_o_subnodes:
                self.level1_o_subnodes[o] = []
            self.level1_o_subnodes[o].append(node)
            
        # 根据子节点创建一级节点
        level1_s_nodes = {}  # s -> node
        level1_o_nodes = {}  # o -> node
        
        # 创建主语模式的一级节点
        for s, subnodes in self.level1_s_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            s_node = FCG_Node(sub=s, level=1, freq=total_freq, feat=avg_feat)
            level1_s_nodes[s] = s_node
            
        # 创建宾语模式的一级节点
        for o, subnodes in self.level1_o_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            o_node = FCG_Node(obj=o, level=1, freq=total_freq, feat=avg_feat)
            level1_o_nodes[o] = o_node
            
        self.l1_nodes.extend(list(level1_s_nodes.values()) + list(level1_o_nodes.values()))
        
        # 4. 构建虚节点
        for sp_node in level2_sp_nodes.values():
            for po_node in level2_po_nodes.values():
                if sp_node.pred == po_node.pred:  # 谓词相同
                    triple = (sp_node.sub, sp_node.pred, po_node.obj)
                    if triple not in level3_real_nodes:  # 不是真实节点
                        # 虚节点特征为相关二级节点的平均
                        virtual_feat = (sp_node.feat + po_node.feat) / 2
                        virtual_node = FCG_Node(sub=sp_node.sub, pred=sp_node.pred, obj=po_node.obj,
                                              level=3, freq=0, feat=virtual_feat)
                        self.l3_nodes.append(virtual_node)
                        # 将虚节点加入到相应的二级节点的子节点列表中
                        self.level2_sp_subnodes[sp_key].append(virtual_node)
                        self.level2_po_subnodes[po_key].append(virtual_node)
                        

    def build_edges(self):
        """构建层级之间的边连接"""
        # 初始化边矩阵
        self.edges_l2_l3 = torch.zeros(len(self.l2_nodes), len(self.l3_nodes))
        self.edges_l1_l2 = torch.zeros(len(self.l1_nodes), len(self.l2_nodes))
        
        # 构建二级节点到三级节点的边
        for i, l2_node in enumerate(self.l2_nodes):
            # 获取二级节点的子节点列表
            subnodes = self.find_subnode(l2_node)
            
            # 计算真实节点最小词频、虚拟节点数量
            min_freq = int(1e9)
            real_count = 0
            for subnode in subnodes:
                if subnode.freq > 0:
                    min_freq = min(min_freq, subnode.freq)
                    real_count += 1
            virtual_count = len(subnodes) - real_count
                    
            # 遍历子节点,设置边权重
            for subnode in subnodes:
                j = self.l3_nodes.index(subnode)
                if subnode.is_virtual:
                    weight = min_freq / (2 * virtual_count) # 最小词频的一半然后均分
                else:
                    weight = subnode.freq / (l2_node.freq + min_freq / 2)
                self.edges_l2_l3[i][j] = weight

        # 构建一级节点到二级节点的边
        for i, l1_node in enumerate(self.l1_nodes):
            subnodes = self.find_subnode(l1_node)
            for subnode in subnodes:
                j = self.l2_nodes.index(subnode)
                self.edges_l1_l2[i][j] = subnode.freq / l1_node.freq
                    

    def find_subnode(self, node: FCG_Node):
        assert node.level != 3
        if node.level == 1 and node.sub is not None:
            return self.level1_s_subnodes.get(node.sub, [])
        elif node.level == 1 and node.obj is not None:
            return self.level1_o_subnodes.get(node.obj, [])
        elif node.level == 2 and node.sub is not None:
            return self.level2_sp_subnodes.get((node.sub, node.pred), [])
        else:
            return self.level2_po_subnodes.get((node.pred, node.obj), [])
        
    def dump_nodes(self):
        """保存FCG的节点和边信息到文件"""
        
        # 1. 保存节点特征
        node_feats = {
            'l1_nodes': [node.feat.tolist() for node in self.l1_nodes],
            'l2_nodes': [node.feat.tolist() for node in self.l2_nodes],
            'l3_nodes': [node.feat.tolist() for node in self.l3_nodes]
        }
        with open(data_path('fcg_nodes.pkl'), 'wb') as f:
            pickle.dump(node_feats, f)
            
        # 2. 保存边信息
        edge_info = {
            'edges_l2_l3': self.edges_l2_l3.tolist(),
            'edges_l1_l2': self.edges_l1_l2.tolist()
        }
        with open(data_path('fcg_edges.pkl'), 'wb') as f:
            pickle.dump(edge_info, f)
            
        # 3. 保存节点元数据
        ind_to_classes, ind_to_predicates = load_info(VG_SGG_DICT_FN)
        metadata = {
            'nodes': []
        }
        
        for nodes in [self.l1_nodes, self.l2_nodes, self.l3_nodes]:
            for i, node in enumerate(nodes):
                metadata['nodes'].append({
                    'freq': int(node.freq),
                    'sub': ind_to_classes[node.sub] if node.sub is not None else None,
                    'pred': ind_to_predicates[node.pred] if node.pred is not None else None,
                    'obj': ind_to_classes[node.obj] if node.obj is not None else None,
                    'level': node.level,
                    'is_virtual': node.is_virtual,
                    'is_sub_pattern': node.is_sub_pattern,
                    'index': i
                })
            
        with open(data_path('fcg_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
if __name__ == '__main__':
    fcg_builder = FCGBuilder(hidden_dim=1024)
    fcg_builder.dump_nodes()