import argparse
import itertools
import os
import pandas as pd
import sys
from random import random

from imitLearningPipelineSharedCode import non_slurm_strategy_ids

parser = argparse.ArgumentParser()
parser.add_argument("--OUTPUT_PATH")
parser.add_argument("--EXP_OUTPUT_PATH")
parser.add_argument("--DATASET_IDS")
parser.add_argument("--STRATEGY_IDS")
parser.add_argument("--NON_SLURM", action="store_true")
parser.add_argument(
    "--AMOUNT_OF_RUNS", type=int, default=1, help="Specifies which dataset to use etc."
)
parser.add_argument("--SLURM_FILE_TO_UPDATE", type=str)
config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


config.DATASET_IDS = [int(item) for item in config.DATASET_IDS.split(",")]
config.STRATEGY_IDS = [int(item) for item in config.STRATEGY_IDS.split(",")]


# this file creates a file called random_ids.csv which contains a list of random_ids for which the resulting data hasn't been created so far
# it contains in a second column the dataset we are dealing with about, and in a third column the id of the AL strategy to use
# the file baseline_comparison.py expects as the parameter "RANDOM_SEED_INDEX" the index for which the random seeds from random_ids.csv should be read

if os.path.isfile(config.EXP_OUTPUT_PATH + "/05_alipy_results.csv"):
    result_df = pd.read_csv(
        config.EXP_OUTPUT_PATH + "/05_alipy_results.csv",
        index_col=None,
        usecols=["dataset_id", "strategy_id", "dataset_random_seed"],
    )
else:
    result_df = pd.DataFrame(
        data=None, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
    )

random_ids_which_should_be_in_evaluation = set(range(0, config.AMOUNT_OF_RUNS))
strategy_ids_which_should_be_in_evaluation = set(config.STRATEGY_IDS)

# remove that strategy_ids which should NOT be run on HPC (slurm), and which can/should not

if config.NON_SLURM:
    strategy_ids_which_should_be_in_evaluation = [
        id
        for id in strategy_ids_which_should_be_in_evaluation
        if id in non_slurm_strategy_ids
    ]
else:
    strategy_ids_which_should_be_in_evaluation = [
        id
        for id in strategy_ids_which_should_be_in_evaluation
        if id not in non_slurm_strategy_ids
    ]


missing_ids = []

for dataset_id, strategy_id, dataset_random_seed in itertools.product(
    config.DATASET_IDS,
    strategy_ids_which_should_be_in_evaluation,
    random_ids_which_should_be_in_evaluation,
    repeat=1,
):
    if (
        len(
            result_df.loc[
                (result_df["dataset_id"] == dataset_id)
                & (result_df["strategy_id"] == strategy_id)
                & (result_df["dataset_random_seed"] == dataset_random_seed)
            ]
        )
        == 0
    ):
        missing_ids.append([dataset_id, strategy_id, dataset_random_seed])


random_seed_df = pd.DataFrame(
    data=missing_ids, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
)
os.makedirs(config.OUTPUT_PATH, exist_ok=True)

output_file = "04_random_seeds_"
if config.NON_SLURM:
    output_file += "_bash.csv"
else:
    output_file += "_slurm.csv"

random_seed_df.to_csv(config.OUTPUT_PATH + "/" + output_file, header=True)


# update slurm/bash file

print(config.SLURM_FILE_TO_UPDATE)
with open(config.SLURM_FILE_TO_UPDATE, "r") as f:
    # replace stuff initially
    # new_content = f.read().replace("XXX", str(len(random_seed_df)))
    if config.SLURM_FILE_TO_UPDATE.endswith(".sh"):
        replace_token = "--N_TASKS "
    else:
        replace_token = "--array 0-"
    new_content = f.read()
    replace_token_position = new_content.find(replace_token)
    end_position = new_content[replace_token_position:].find("\n")
    if end_position == -1:
        end_position = len(new_content)
    else:
        end_position += replace_token_position

    to_be_replaced = new_content[replace_token_position:end_position]

    new_content = new_content.replace(
        to_be_replaced, replace_token + str(len(random_seed_df))
    )

with open(config.SLURM_FILE_TO_UPDATE, "w") as f:
    f.write(new_content)
