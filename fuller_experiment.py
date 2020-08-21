import pandas as pd
import os
import subprocess
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
from active_learning.experiment_setup_lib import standard_config

# example:
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 means that we call it once with this, and once with the default values, and compare both
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 3 means that we call it once with 2, and once with 3, and once with default values
# python fuller_experiment.py --TRAIN_HYPERCUBE means that we call it once with, and once without
# python fuller_experiment --TRAIN_HYPERCUBE --TEST_OLD_SYNTHETIC_PARAMS means that we have a 2x2 comparisn

config = standard_config(
    [
        (["--RANDOM_SEED"], {"default": 1}),
        (["--LOG_FILE"], {"default": "log.txt"}),
        (["--OUTPUT_DIRECTORY"], {"default": "tmp"}),
        (["--TRAIN_VARIABLE_DATASET"], {"default": "default"}),
        (["--TRAIN_NR_LEARNING_SAMPLES"], {"default": "default"}),
        (["--TRAIN_REPRESENTATIVE_FEATURES"], {"default": "default"}),
        (["--TRAIN_AMOUNT_OF_FEATURES"], {"default": "default"}),
        (["--TRAIN_HYPERCUBE"], {"default": "default"}),
        (["--TRAIN_OLD_SYNTHETIC_PARAMS"], {"default": "default"}),
        (["--TEST_VARIABLE_DATASET"], {"default": "default"}),
        (["--TEST_NR_LEARNING_SAMPLES"], {"default": "default"}),
        (["--TEST_REPRESENTATIVE_FEATURES"], {"default": "default"}),
        (["--TEST_AMOUNT_OF_FEATURES"], {"default": "default"}),
        (["--TEST_HYPERCUBE"], {"default": "default"}),
        (["--TEST_OLD_SYNTHETIC_PARAMS"], {"default": "default"}),
        (["--TEST_COMPARISONS"], {"default": "default"}),
    ],
    standard_args=False,
)


cli_commands = {0: "python full_experiment.py"}

arguments = []

for k, v in vars(config).items():
    if k in ["RANDOM_SEED", "LOG_FILE", "OUTPUT_DIRECTORY"]:
        cli_commands[0] += " --" + k + " " + str(v)
        continue

    if v == "default":
        continue

    splitted_inputs = str(v).split(",")
    if len(splitted_inputs) == 1:
        arguments.append((k, [None, splitted_inputs[0]]))
    elif len(splitted_inputs) > 1:
        arguments.append((k, splitted_inputs))

to_be_removed_later = cli_commands[0]

for k, argument in arguments:
    offset = len(cli_commands)

    # duplicate list
    for i in range(0, len(cli_commands) * len(argument)):
        cli_commands[i] = cli_commands[i % offset]

    for i, a in enumerate(argument):
        for j in range(offset * i, offset * i + int(len(cli_commands) / len(argument))):
            if a == "True":
                cli_commands[j] += " --" + k
            else:
                cli_commands[j] += " --" + k + " " + str(a)


# compute FINAL_PICTURE argument

for cli_command in cli_commands.values():

    FINAL_PICTURE = (
        cli_command.replace(to_be_removed_later, "")
        .replace(" ", "_")
        .replace("--", "")
        .replace("None", "False")
    )
    cli_command += " --FINAL_PICTURE " + config.OUTPUT_DIRECTORY + "/" + FINAL_PICTURE
    # remove all None things
    for k, v in arguments:
        if None in v:
            #  print(k, str(v))
            cli_command = cli_command.replace("--" + k + " None ", "")
    print(cli_command)
    os.system(cli_command)
