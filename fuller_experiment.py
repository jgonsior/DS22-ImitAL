import os

from active_learning.experiment_setup_lib import standard_config

# example:
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 means that we call it once with this, and once with the default values, and compare both
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 3 means that we call it once with 2, and once with 3, and once with default values
# python fuller_experiment.py --TRAIN_HYPERCUBE means that we call it once with, and once without
# python fuller_experiment --TRAIN_HYPERCUBE --TEST_NEW_SYNTHETIC_PARAMS means that we have a 2x2 comparisn

config = standard_config(
    [
        (["--RANDOM_SEED"], {"default": 1}),
        (["--LOG_FILE"], {"default": "log.txt"}),
        (["--OUTPUT_DIRECTORY"], {"default": "tmp/"}),
        (["--NR_QUERIES_PER_ITERATION"], {"default": "default"}),
        (["--USER_QUERY_BUDGET_LIMIT"], {"default": "default"}),
        (["--TRAIN_CLASSIFIER"], {"default": "default"}),
        (["--TRAIN_VARIABLE_DATASET"], {"default": "default"}),
        (["--TRAIN_NR_LEARNING_SAMPLES"], {"default": "default"}),
        (["--TRAIN_AMOUNT_OF_FEATURES"], {"default": "default"}),
        (["--TRAIN_HYPERCUBE"], {"default": "default"}),
        (["--TRAIN_NEW_SYNTHETIC_PARAMS"], {"default": "default"}),
        (["--TRAIN_CONVEX_HULL_SAMPLING"], {"default": "default"}),
        (["--TRAIN_VARIANCE_BOUND"], {"default": "default"}),
        (["--TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED"], {"default": "default"}),
        (["--TRAIN_GENERATE_NOISE"], {"default": "default"}),
        (["--TRAIN_NO_DIFF_FEATURES"], {"default": "default"}),
        (["--TRAIN_STATE_DIFF_PROBAS"], {"default": "default"}),
        (["--TRAIN_STATE_ARGSECOND_PROBAS"], {"default": "default"}),
        (["--TRAIN_STATE_ARGTHIRD_PROBAS"], {"default": "default"}),
        (["--TRAIN_STATE_DISTANCES"], {"default": "default"}),
        (["--TRAIN_STATE_NO_LRU_WEIGHTS"], {"default": "default"}),
        (["--TRAIN_STATE_LRU_AREAS_LIMIT"], {"default": "default"}),
        (["--TEST_VARIABLE_DATASET"], {"default": "default"}),
        (["--TEST_NR_LEARNING_SAMPLES"], {"default": "default"}),
        (["--TEST_AMOUNT_OF_FEATURES"], {"default": "default"}),
        (["--TEST_HYPERCUBE"], {"default": "default"}),
        (["--TEST_NEW_SYNTHETIC_PARAMS"], {"default": "default"}),
        (["--TEST_COMPARISONS"], {"default": "default"}),
        (["--TEST_CONVEX_HULL_SAMPLING"], {"default": "default"}),
        (["--TEST_CLASSIFIER"], {"default": "default"}),
        (["--TEST_GENERATE_NOISE"], {"default": "default"}),
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
FINAL_PICTURE_FOLDER = "plots"

for k, argument in arguments:
    FINAL_PICTURE_FOLDER += "_" + k

for cli_command in cli_commands.values():
    cli_args = cli_command.split(" --")[4:]

    FINAL_PICTURE = ""
    for cli_arg in cli_args:
        if not " " in cli_arg:
            FINAL_PICTURE += "_True"
        else:
            FINAL_PICTURE += "_" + cli_arg.split(" ")[1].replace("None", "False")

    cli_command += (
        " --FINAL_PICTURE "
        + config.OUTPUT_DIRECTORY
        + FINAL_PICTURE_FOLDER
        + "/"
        + FINAL_PICTURE
    )
    os.makedirs(config.OUTPUT_DIRECTORY + FINAL_PICTURE_FOLDER, exist_ok=True)
    # remove all None things
    for k, v in arguments:
        if None in v:
            #  print(k, str(v))
            cli_command = cli_command.replace("--" + k + " None ", "")
    BASE_PARAM_STRING = cli_command.split(" --")[4:-1]
    BASE_PARAM_STRING = "#".join(BASE_PARAM_STRING).replace(" ", "_")
    if BASE_PARAM_STRING == "":
        BASE_PARAM_STRING = "DEFAULT"
    cli_command += " --BASE_PARAM_STRING " + BASE_PARAM_STRING
    print(cli_command)
    os.system(cli_command)
