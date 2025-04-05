import torch
import torch.nn.functional as F
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)  # 编译 Cython 模块

from model.feature.fcg_net import FCGNet


def fcg_loss(fcg_dist, labels):
    """
    FCG损失函数
    """
    return F.nll_loss(torch.log(fcg_dist + 1e-10), labels)


def test_fcg():
    """
    测试FCGNet的前向传播和反向传播
    使用随机生成的张量模拟输入
    """
    torch.autograd.set_detect_anomaly(True)

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 定义测试参数
    hidden_dim = 1024  # 隐藏层维度
    num_entities = 20  # 实体数量
    num_relations = 15  # 关系数量
    learning_rate = 0.001  # 学习率
    num_epochs = 3  # 训练轮数

    # 创建FCGNet实例
    fcg_net = FCGNet(hidden_dim=hidden_dim).cuda()
    fcg_net.train()

    # 创建优化器
    optimizer = torch.optim.Adam(fcg_net.parameters(), lr=learning_rate)

    # 1. 生成随机输入
    # 关系索引: 形状为(num_relations, 2)的张量，表示<s,o>对
    rel_inds = torch.randint(0, num_entities, (num_relations, 2), dtype=torch.long).cuda()

    # 实体概率: 形状为(num_entities, 151)的张量，表示实体类别概率分布
    ent_probs = torch.rand(num_entities, 151).cuda()
    ent_probs = ent_probs / ent_probs.sum(dim=1, keepdim=True)  # 归一化

    # 关系视觉特征: 形状为(num_relations, hidden_dim)的张量
    vr = torch.rand(num_relations, hidden_dim).cuda()

    # 生成随机标签用于计算损失
    labels = torch.randint(0, 51, (num_relations,), dtype=torch.long).cuda()

    print(f"开始训练，总共 {num_epochs} 轮...")

    # 多轮训练
    for epoch in range(num_epochs):
        print(f"\n【第 {epoch+1}/{num_epochs} 轮训练】")

        # 2. 前向传播
        print("开始前向传播...")
        pred_cls_score = fcg_net(rel_inds, ent_probs, vr)
        print(f"前向传播完成，输出形状: {pred_cls_score.shape}")

        # 3. 计算损失
        loss = fcg_loss(pred_cls_score, labels)
        print(f"损失值: {loss.item():.4f}")

        # 4. 梯度清零
        optimizer.zero_grad()

        # 5. 反向传播
        print("开始反向传播...")
        loss.backward()
        print("反向传播完成")

        # 6. 参数更新
        print("开始更新参数...")
        optimizer.step()
        print(f"参数已更新，学习率: {learning_rate}")

    print("\n训练完成！")

if __name__ == "__main__":
    print("【FCGNet 前向与反向传播测试】")
    test_fcg()
    print("【测试完成】")
