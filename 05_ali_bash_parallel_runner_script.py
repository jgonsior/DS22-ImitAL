import os
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--N_TASKS", type=int)
parser.add_argument("--N_PARALLLEL_JOBS", default=20, type=int)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


def run_code(i):
    cli = (
        "python 05_alipy_eva.py --OUTPUT_PATH ../datasets/ali_non_slurm --INDEX "
        + str(i)
    )
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=config.N_PARALLEL_JOBS):
    Parallel()(delayed(run_code)(i) for i in range(config.N_TASKS))
