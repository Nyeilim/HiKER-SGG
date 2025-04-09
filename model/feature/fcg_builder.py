import pickle

import numpy as np
import torch

from config import NODE_EMBEDDING, EDGE_MATRIX


class FCGNode:
    """FCG图中的节点类，用于统一管理节点信息"""

    def __init__(self, sub=None, pred=None, obj=None, level=None, freq=None, feat=None):
        self.freq = freq  # 词频，样本数量
        self.sub = sub  # 主语
        self.pred = pred  # 谓词
        self.obj = obj  # 宾语
        self.level = level  # 层级(1,2,3)
        self.feat = feat  # 节点特征
        self.idx = -1  # 节点下标，在节点构建的最后时刻填充

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
        self.edge_matrix = np.load(EDGE_MATRIX)  # 151x151x51

        # 初始化正交投影矩阵，构建映射 900->1024
        input_dim = self.emb_ent.shape[1] * 2 + self.emb_pred.shape[1]
        output_dim = hidden_dim
        M, _ = torch.linalg.qr(torch.randn(output_dim, input_dim))  # 生成正交基
        self.proj_matrix = M.T * np.sqrt(output_dim / input_dim)  # 缩放保持方差
        self.proj_matrix.requires_grad_(False)  # 禁用梯度

        # 按层级存储节点
        self.l1_nodes = []  # 第一级节点
        self.l2_nodes = []  # 第二级节点
        self.l3_nodes = []  # 第三级节点(包括真实节点和虚节点)

        # 存储各层级节点的子节点
        self.level2_sp_subnodes = {}  # (s,p) -> [node1, node2, ...]
        self.level2_po_subnodes = {}  # (p,o) -> [node1, node2, ...]
        self.level1_s_subnodes = {}  # s -> [node1, node2, ...]
        self.level1_o_subnodes = {}  # o -> [node1, node2, ...]

        # 构建字典
        self.level3_so_nodes = {}  # (s,o) -> [node1, node2, ...]

        # 边权重
        self.edges_l2_l3 = None
        self.edges_l1_l2 = None

        # 构建节点和边
        self.build_nodes()
        self.build_edges()

        self.data_check()

    def build_nodes(self):
        """构建FCG的所有节点"""
        # 1. 构建第三级真实节点
        for s in range(151):
            for o in range(151):
                for p in range(51):
                    freq = self.edge_matrix[s, o, p]
                    if freq > 0:
                        # 计算节点特征
                        feat = torch.cat([
                            torch.tensor(self.emb_ent[s], dtype=torch.float32),
                            torch.tensor(self.emb_pred[p], dtype=torch.float32),
                            torch.tensor(self.emb_ent[o], dtype=torch.float32)
                        ]) @ self.proj_matrix  # (900,) @ (900,1024) -> (1024,)

                        node = FCGNode(sub=s, pred=p, obj=o, level=3, freq=freq, feat=feat)
                        self.l3_nodes.append(node)

                        so_key = (s, o)
                        self.add_node(so_key, self.level3_so_nodes, node)

        # 2. 构建第二级节点
        # 先收集每个二级节点的所有子节点
        for node in self.l3_nodes:
            # 收集(s,p)模式的子节点
            sp_key = (node.sub, node.pred)
            self.add_node(sp_key, self.level2_sp_subnodes, node)

            # 收集(p,o)模式的子节点
            po_key = (node.pred, node.obj)
            self.add_node(po_key, self.level2_po_subnodes, node)

        # 根据子节点创建二级节点
        level2_sp_nodes = {}  # (s,p) -> node
        level2_po_nodes = {}  # (p,o) -> node

        # 创建(s,p)模式的二级节点
        for sp_key, subnodes in self.level2_sp_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            sp_node = FCGNode(sub=sp_key[0], pred=sp_key[1], level=2, freq=total_freq, feat=avg_feat)
            level2_sp_nodes[sp_key] = sp_node

        # 创建(p,o)模式的二级节点
        for po_key, subnodes in self.level2_po_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            po_node = FCGNode(pred=po_key[0], obj=po_key[1], level=2, freq=total_freq, feat=avg_feat)
            level2_po_nodes[po_key] = po_node

        self.l2_nodes.extend(list(level2_sp_nodes.values()) + list(level2_po_nodes.values()))

        # 3. 构建第一级节点
        # 先收集每个一级节点的所有子节点
        for node in level2_sp_nodes.values():
            # 收集主语模式的子节点
            s = node.sub
            self.add_node(s, self.level1_s_subnodes, node)

        for node in level2_po_nodes.values():
            # 收集宾语模式的子节点
            o = node.obj
            self.add_node(o, self.level1_o_subnodes, node)

        # 根据子节点创建一级节点
        level1_s_nodes = {}  # s -> node
        level1_o_nodes = {}  # o -> node

        # 创建主语模式的一级节点
        for s, subnodes in self.level1_s_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            s_node = FCGNode(sub=s, level=1, freq=total_freq, feat=avg_feat)
            level1_s_nodes[s] = s_node

        # 创建宾语模式的一级节点
        for o, subnodes in self.level1_o_subnodes.items():
            total_freq = sum(node.freq for node in subnodes)
            avg_feat = sum(node.feat for node in subnodes) / len(subnodes)
            o_node = FCGNode(obj=o, level=1, freq=total_freq, feat=avg_feat)
            level1_o_nodes[o] = o_node

        self.l1_nodes.extend(list(level1_s_nodes.values()) + list(level1_o_nodes.values()))

        # # 4. 构建虚节点，这步会引入大量的虚拟节点，导致内存溢出。所以我先关闭这个特性
        # for sp_node in level2_sp_nodes.values():
        #     for po_node in level2_po_nodes.values():
        #         if sp_node.pred == po_node.pred:  # 谓词相同
        #             triple = (sp_node.sub, sp_node.pred, po_node.obj)
        #             sp_key = (sp_node.sub, sp_node.pred)
        #             po_key = (po_node.pred, po_node.obj)
        #             if triple not in level3_real_nodes:  # 不是真实节点
        #                 # 虚节点特征为相关二级节点的平均
        #                 virtual_feat = (sp_node.feat + po_node.feat) / 2
        #                 virtual_node = FCGNode(sub=sp_node.sub, pred=sp_node.pred, obj=po_node.obj,
        #                                        level=3, freq=0, feat=virtual_feat)
        #                 self.l3_nodes.append(virtual_node)
        #                 # 将虚节点加入到相应的二级节点的子节点列表中
        #                 self.level2_sp_subnodes[sp_key].append(virtual_node)
        #                 self.level2_po_subnodes[po_key].append(virtual_node)

        # 所有节点构建完成，填充下标
        for i, node in enumerate(self.l1_nodes):
            node.idx = i
        for i, node in enumerate(self.l2_nodes):
            node.idx = i
        for i, node in enumerate(self.l3_nodes):
            node.idx = i

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
                j = subnode.idx
                if subnode.is_virtual:
                    weight = min_freq / (2 * virtual_count)  # 最小词频的一半然后均分
                else:
                    weight = subnode.freq / (l2_node.freq + min_freq / 2)
                self.edges_l2_l3[i][j] = weight

        # 构建一级节点到二级节点的边
        for i, l1_node in enumerate(self.l1_nodes):
            subnodes = self.find_subnode(l1_node)
            for subnode in subnodes:
                j = subnode.idx
                self.edges_l1_l2[i][j] = subnode.freq / l1_node.freq

    def find_subnode(self, node: FCGNode):
        assert node.level != 3
        if node.level == 1 and node.sub is not None:
            return self.level1_s_subnodes.get(node.sub, [])
        elif node.level == 1 and node.obj is not None:
            return self.level1_o_subnodes.get(node.obj, [])
        elif node.level == 2 and node.sub is not None:
            return self.level2_sp_subnodes.get((node.sub, node.pred), [])
        elif node.level == 2 and node.obj is not None:
            return self.level2_po_subnodes.get((node.pred, node.obj), [])
        else:
            raise ValueError(f"This don't have subnodes: {node}")

    @staticmethod
    def add_node(key, key2nodes, node):
        if key not in key2nodes:
            key2nodes[key] = []
        key2nodes[key].append(node)

    def has_sample(self, sub, obj):
        nodes = self.get_sample(sub, obj)
        return len(nodes) > 0

    def get_sample(self, sub, obj):
        # 根据 s,o 获取 l3_nodes 节点列表
        so_key = (sub, obj)
        ret = []
        if so_key in self.level3_so_nodes:
            ret = self.level3_so_nodes[so_key]
        return ret

    def data_check(self):
        # 看看各个子列表的的总和加起来是不是等于节点总数
        counts = {
            'level2_sp_subnodes': sum(len(nodes) for nodes in self.level2_sp_subnodes.values()),
            'level2_po_subnodes': sum(len(nodes) for nodes in self.level2_po_subnodes.values()),
            'level1_s_subnodes': sum(len(nodes) for nodes in self.level1_s_subnodes.values()),
            'level1_o_subnodes': sum(len(nodes) for nodes in self.level1_o_subnodes.values())
        }
        assert counts['level2_sp_subnodes'] == counts['level2_po_subnodes'] == len(self.l3_nodes)
        assert counts['level1_s_subnodes'] + counts['level1_o_subnodes'] == len(self.l2_nodes)


# 该文件作为模块导入时，下面这行代码不会被执行
if __name__ == '__main__':
    fcg_builder = FCGBuilder(hidden_dim=1024)
    sample = fcg_builder.get_sample(1, 2)
