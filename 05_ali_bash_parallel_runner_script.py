import argparse
import multiprocessing
import os
import sys
from joblib import Parallel, delayed, parallel_backend

parser = argparse.ArgumentParser()
parser.add_argument("--N_TASKS", type=int)
parser.add_argument("--N_PARALLEL_JOBS", type=int)
parser.add_argument("--OUTPUT_PATH")
parser.add_argument("--DATASETS_DIR")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


def run_code(i):
    cli = (
        "python 05_alipy_eva.py --OUTPUT_PATH "
        + config.OUTPUT_PATH
        + " --INDEX "
        + str(i)
        + " --DATASETS_DIR "
        + config.DATASETS_DIR
        + " --RANDOM_SEEDS_INPUT_FILE "
        + config.OUTPUT_PATH
        + "/04_random_seeds__bash.csv"
    )
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=config.N_PARALLEL_JOBS):
    Parallel()(delayed(run_code)(i) for i in range(config.N_TASKS))
