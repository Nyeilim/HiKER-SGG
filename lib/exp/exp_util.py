import json

best_matrices_file = '/output/data/best_matrices.json'


def save_best_matrices(matrices_list, nc_matrices_list):
    sum_list = [sum(row) for row in matrices_list]
    max_value_index = find_max_value_index(sum_list)

    nc_sum_list = [sum(row) for row in nc_matrices_list]
    nc_max_value_index = find_max_value_index(nc_sum_list)

    matrices_data = {
        "best_mr": matrices_list[max_value_index],
        "best_mr_epoch": max_value_index,
        "best_nc_mr": nc_matrices_list[nc_max_value_index],
        "best_nc_mr_epoch": nc_max_value_index
    }

    json_string = json.dumps(matrices_data, ensure_ascii=False)
    with open(best_matrices_file, 'w', encoding='utf-8') as file:
        file.write(json_string)


def load_best_matrices():
    with open(best_matrices_file, 'r', encoding='utf-8') as file:
        matrices_data = json.load(file)

    return matrices_data


def find_max_value_index(_list):
    max_value = max(_list)
    max_index = _list.index(max_value)
    return max_index
