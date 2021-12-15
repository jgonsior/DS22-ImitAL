import argparse
import multiprocessing
import os
import sys
from joblib import Parallel, delayed, parallel_backend

parser = argparse.ArgumentParser()
parser.add_argument("--N_TASKS", type=int)
parser.add_argument("--N_PARALLEL_JOBS", type=int)
parser.add_argument("--OUTPUT_PATH")
parser.add_argument("--DATASETS_PATH")
parser.add_argument("--RANDOM_SEEDS_PATH")

parser.add_argument("--EXCLUDING", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DIFF_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGFIRST_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGSECOND_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGTHIRD_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES_LAB", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES_UNLAB", action="store_true")
parser.add_argument("--EXCLUDING_STATE_PREDICTED_CLASS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_PREDICTED_UNITY", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES", action="store_true")
parser.add_argument("--EXCLUDING_STATE_UNCERTAINTIES", action="store_true")
parser.add_argument("--EXCLUDING_STATE_INCLUDE_NR_FEATURES", action="store_true")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


STATE_APPENDIX = ""
if config.EXCLUDING_STATE_DISTANCES_LAB:
    STATE_APPENDIX += " --EXCLUDING_STATE_DISTANCES_LAB"
if config.EXCLUDING_STATE_DISTANCES_UNLAB:
    STATE_APPENDIX += " --EXCLUDING_STATE_DISTANCES_UNLAB"
if config.EXCLUDING_STATE_PREDICTED_CLASS:
    STATE_APPENDIX += " --EXCLUDING_STATE_PREDICTED_CLASS"
if config.EXCLUDING_STATE_PREDICTED_UNITY:
    STATE_APPENDIX += " --EXCLUDING_STATE_PREDICTED_UNITY"
if config.EXCLUDING_STATE_ARGFIRST_PROBAS:
    STATE_APPENDIX += " --EXCLUDING_STATE_ARGFIRST_PROBAS"
if config.EXCLUDING_STATE_ARGSECOND_PROBAS:
    STATE_APPENDIX += " --EXCLUDING_STATE_ARGSECOND_PROBAS"
if config.EXCLUDING_STATE_ARGTHIRD_PROBAS:
    STATE_APPENDIX += " --EXCLUDING_STATE_ARGTHIRD_PROBAS"
if config.EXCLUDING_STATE_DIFF_PROBAS:
    STATE_APPENDIX += " --EXCLUDING_STATE_DIFF_PROBAS"
if config.EXCLUDING_STATE_DISTANCES:
    STATE_APPENDIX += " --EXCLUDING_STATE_DISTANCES"
if config.EXCLUDING_STATE_UNCERTAINTIES:
    STATE_APPENDIX += " --EXCLUDING_STATE_UNCERTAINTIES"
if config.EXCLUDING_STATE_INCLUDE_NR_FEATURES:
    STATE_APPENDIX += " --EXCLUDING_STATE_INCLUDE_NR_FEATURES"

if STATE_APPENDIX == "":
    STATE_APPENDIX = "None"
else:
    STATE_APPENDIX = STATE_APPENDIX[3:]


def run_code(i):
    cli = (
        "python 05_alipy_eva.py --OUTPUT_PATH "
        + config.OUTPUT_PATH
        + " --INDEX "
        + str(i)
        + " --DATASETS_PATH "
        + config.DATASETS_PATH
        + STATE_APPENDIX
        + " --RANDOM_SEEDS_INPUT_FILE "
        + config.RANDOM_SEEDS_PATH
        + "/04_random_seeds__slurm.csv"
    )
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=config.N_PARALLEL_JOBS):
    Parallel()(delayed(run_code)(i) for i in range(config.N_TASKS))
