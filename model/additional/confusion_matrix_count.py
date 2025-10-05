import json
import sys
import numpy as np

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
from config import data_path, VG_SGG_DICT_FN

def analyze_confusion_matrix():
    """
    读取混淆矩阵文件，统计每个谓词的错误数量、总预测数和错误率
    """
    # 读取混淆矩阵文件
    confusion_matrix = np.load(data_path('confusion_matrix.npy'))
    print(f"混淆矩阵形状: {confusion_matrix.shape}")
    
    # 确保矩阵是 51x51 的
    if confusion_matrix.shape != (51, 51):
        print(f"错误: 混淆矩阵形状应为 (51, 51)，实际为 {confusion_matrix.shape}")
        return
    
    # 加载谓词索引到名称的映射
    with open(VG_SGG_DICT_FN) as f:
        vg_dict = json.load(f)
    ind_to_predicates = vg_dict['idx_to_predicate']  # 1-indexed

    # 添加空关系在索引0位置
    predicate_names = ['__background__'] + [ind_to_predicates[str(i)] for i in range(1, 51)]
    
    # 统计每个谓词的错误数量、总预测数和错误率
    predicate_stats = []
    
    for i in range(1, 51):  # 从1开始，跳过空关系(索引0)
        # 对于谓词i:
        # - 总预测数 = 混淆矩阵第i行的总和
        # - 正确预测数 = 混淆矩阵第i行第i列的值
        # - 错误数量 = 总预测数 - 正确预测数
        # - 错误率 = 错误数量 / 总预测数
        
        total_predictions = np.sum(confusion_matrix[i, :])
        correct_predictions = confusion_matrix[i, i]
        error_count = total_predictions - correct_predictions
        
        # 避免除零错误
        if total_predictions > 0:
            error_rate = error_count / total_predictions
        else:
            error_rate = 0.0
            
        predicate_stats.append({
            'index': i,
            'name': predicate_names[i],
            'total_predictions': int(total_predictions),
            'error_count': int(error_count),
            'error_rate': error_rate
        })
    
    # 按错误率从高到低排序
    predicate_stats.sort(key=lambda x: x['error_rate'], reverse=True)
    
    # 打印结果
    print("\n谓词错误率统计 (按错误率从高到低排序):")
    print("-" * 80)
    print(f"{'排名':<4} {'谓词名称':<20} {'总预测数':<10} {'错误数':<8} {'错误率':<10}")
    print("-" * 80)
    
    for rank, stat in enumerate(predicate_stats, 1):
        print(f"{rank:<4} {stat['name']:<20} {stat['total_predictions']:<10} "
              f"{stat['error_count']:<8} {stat['error_rate']:.4f}")

if __name__ == "__main__":
    analyze_confusion_matrix()