import json
import os
import glob
from config import data_path

best_matrices_file = data_path('best_matrices.json')

# 这边 List 拿到的数据长这样：[{'R@100': 0.32365809238988463, 'R@20': 0.275271453097786, 'R@50': 0.3127690749291039}, {...}]
def save_best_matrices(matrices_list, nc_matrices_list, task_type='predcls'):
    """
    保存最佳模型信息，支持不同任务类型
    :param matrices_list: 每个epoch的mean recall列表
    :param nc_matrices_list: 每个epoch的mean recall without constraint列表
    :param task_type: 任务类型，'predcls' 或 'sgcls'
    """
    sum_list = [sum([row['R@100'], row['R@50'], row['R@20']]) for row in matrices_list]
    
    # 获取前两个最佳模型的索引
    sorted_indices = sorted(range(len(sum_list)), key=lambda i: sum_list[i], reverse=True)
    best_epoch = sorted_indices[0]
    second_best_epoch = sorted_indices[1] if len(sorted_indices) > 1 else best_epoch

    nc_sum_list = [sum([row['R@100'], row['R@50'], row['R@20']]) for row in nc_matrices_list]
    nc_max_value_index = find_max_value_index(nc_sum_list)

    # 读取现有的最佳模型信息（如果存在）
    try:
        with open(best_matrices_file, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    # 为当前任务类型保存最佳模型信息
    task_key = f"{task_type}_best_mr"
    task_epoch_key = f"{task_type}_best_mr_epoch"
    task_second_key = f"{task_type}_second_best_mr"
    task_second_epoch_key = f"{task_type}_second_best_mr_epoch"
    task_nc_key = f"{task_type}_best_nc_mr"
    task_nc_epoch_key = f"{task_type}_best_nc_mr_epoch"

    matrices_data = {
        **existing_data,  # 保留其他任务的信息
        task_key: matrices_list[best_epoch],
        task_epoch_key: best_epoch,
        task_second_key: matrices_list[second_best_epoch],
        task_second_epoch_key: second_best_epoch,
        task_nc_key: nc_matrices_list[nc_max_value_index],
        task_nc_epoch_key: nc_max_value_index
    }

    json_string = json.dumps(matrices_data, ensure_ascii=False)
    with open(best_matrices_file, 'w', encoding='utf-8') as file:
        file.write(json_string)


def load_best_matrices(task_type='predcls'):
    """
    加载最佳模型信息，支持不同任务类型
    :param task_type: 任务类型，'predcls' 或 'sgcls'
    :return: 最佳模型信息字典
    """
    with open(best_matrices_file, 'r', encoding='utf-8') as file:
        matrices_data = json.load(file)

    # 使用任务特定的键名
    task_key = f"{task_type}_best_mr"
    task_epoch_key = f"{task_type}_best_mr_epoch"
    task_second_key = f"{task_type}_second_best_mr"
    task_second_epoch_key = f"{task_type}_second_best_mr_epoch"
    task_nc_key = f"{task_type}_best_nc_mr"
    task_nc_epoch_key = f"{task_type}_best_nc_mr_epoch"

    return {
        'best_mr': matrices_data.get(task_key),
        'best_mr_epoch': matrices_data.get(task_epoch_key),
        'second_best_mr': matrices_data.get(task_second_key),
        'second_best_mr_epoch': matrices_data.get(task_second_epoch_key),
        'best_nc_mr': matrices_data.get(task_nc_key),
        'best_nc_mr_epoch': matrices_data.get(task_nc_epoch_key)
    }


def find_max_value_index(_list):
    max_value = max(_list)
    max_index = _list.index(max_value)
    return max_index


def cleanup_model_files(task_type='predcls', keep_epochs=None):
    """
    清理不需要的模型文件，只保留指定的epoch
    :param task_type: 任务类型，'predcls' 或 'sgcls'
    :param keep_epochs: 需要保留的epoch列表
    """
    if keep_epochs is None:
        keep_epochs = []
    
    save_dir = data_path(f"checkpoints/kern_{task_type}/hikersgg_{task_type}_train")
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist, skipping cleanup")
        return
    
    # 获取所有模型文件
    model_files = glob.glob(os.path.join(save_dir, "vgrel-*.tar"))
    
    cleaned_count = 0
    for model_file in model_files:
        # 从文件名中提取epoch号
        filename = os.path.basename(model_file)
        try:
            epoch = int(filename.replace("vgrel-", "").replace(".tar", ""))
            if epoch not in keep_epochs:
                os.remove(model_file)
                cleaned_count += 1
                print(f"Deleted model file: {model_file}")
        except ValueError:
            print(f"Cannot parse epoch from filename: {filename}")
    
    print(f"Cleanup completed, deleted {cleaned_count} model files")
