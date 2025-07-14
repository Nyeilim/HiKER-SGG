import sys
sys.path.append("/output/HiKER-SGG/")

from run.util import test_flow

if __name__ == "__main__":
    test_flow(task_type='sgcls')