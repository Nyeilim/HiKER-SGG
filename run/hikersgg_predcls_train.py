import sys
sys.path.append("/output/HiKER-SGG/")

from run.util import train_flow

if __name__ == "__main__":
    train_flow(task_type='predcls')
