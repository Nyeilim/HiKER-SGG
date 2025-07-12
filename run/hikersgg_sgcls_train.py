import sys
sys.path.append("/output/HiKER-SGG/")

from utils import train_flow

if __name__ == "__main__":
    train_flow(task_type='sgcls') 