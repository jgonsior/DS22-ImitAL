import math
import multiprocessing
import os
from pathlib import Path
import time
import pandas as pd
from joblib import Parallel, delayed

from active_learning.experiment_setup_lib import standard_config

config, parser = standard_config(
    [
        (["--RANDOM_SEED"], {"default": 1, "type": int}),
        (["--LOG_FILE"], {"default": "log.txt"}),
        (["--OUTPUT_DIRECTORY"], {"default": "/tmp"}),
        (["--BASE_PARAM_STRING"], {"default": "default"}),
        (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 1}),
        (["--USER_QUERY_BUDGET_LIMIT"], {"type": int, "default": 50}),
        (["--TRAIN_CLASSIFIER"], {"default": "MLP"}),
        (["--TRAIN_VARIABLE_DATASET"], {"action": "store_false"}),
        (["--TRAIN_NR_LEARNING_SAMPLES"], {"type": int, "default": 3000}),
        (["--TRAIN_AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
        (["--TRAIN_VARIANCE_BOUND"], {"type": int, "default": 1}),
        (["--TRAIN_HYPERCUBE"], {"action": "store_true"}),
        (["--TRAIN_NEW_SYNTHETIC_PARAMS"], {"action": "store_false"}),
        (["--TRAIN_CONVEX_HULL_SAMPLING"], {"action": "store_false"}),
        (["--TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED"], {"action": "store_false"}),
        (["--TRAIN_GENERATE_NOISE"], {"action": "store_true"}),
        (["--TRAIN_STATE_DIFF_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_ARGSECOND_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_ARGTHIRD_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_DISTANCES"], {"action": "store_true"}),
        (["--TRAIN_STATE_NO_LRU_WEIGHTS"], {"action": "store_true"}),
        (["--TRAIN_STATE_LRU_AREAS_LIMIT"], {"type": int, "default": 0}),
        (["--TEST_VARIABLE_DATASET"], {"action": "store_false"}),
        (["--TEST_NR_LEARNING_SAMPLES"], {"type": int, "default": 100}),
        (["--TEST_AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
        (["--TEST_HYPERCUBE"], {"action": "store_true"}),
        (["--TEST_NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
        (["--TEST_CONVEX_HULL_SAMPLING"], {"action": "store_false"}),
        (["--TEST_CLASSIFIER"], {"default": "MLP"}),
        (["--TEST_GENERATE_NOISE"], {"action": "store_false"}),
        (
            ["--TEST_COMPARISONS"],
            {"nargs": "+", "default": ["random", "uncertainty_max_margin"]},
        ),
        (["--FINAL_PICTURE"], {"default": ""}),
    ],
    standard_args=False,
    return_argparse=True,
)
splitted_base_param_string = config.BASE_PARAM_STRING.split("#")
train_base_param_string = "#".join(
    [x for x in splitted_base_param_string if not x.startswith("TEST_")]
)
test_base_param_string = "#".join(
    [x for x in splitted_base_param_string if not x.startswith("TRAIN_")]
)

if train_base_param_string == "":
    train_base_param_string = "DEFAULT"
if test_base_param_string == "":
    test_base_param_string = "DEFAULT"

PARENT_OUTPUT_DIRECTORY = config.OUTPUT_DIRECTORY

print("train_base_param_string" + train_base_param_string)
print("test_base_param_string" + test_base_param_string)
print(config.BASE_PARAM_STRING)
print()

params = {
    "USER_QUERY_BUDGET_LIMIT": config.USER_QUERY_BUDGET_LIMIT,
    "VARIABLE_DATASET": config.TRAIN_VARIABLE_DATASET,
    "NR_LEARNING_SAMPLES": config.TRAIN_NR_LEARNING_SAMPLES,
    "AMOUNT_OF_FEATURES": config.TRAIN_AMOUNT_OF_FEATURES,
    "HYPERCUBE": config.TRAIN_HYPERCUBE,
    "NEW_SYNTHETIC_PARAMS": config.TRAIN_NEW_SYNTHETIC_PARAMS,
    "CONVEX_HULL_SAMPLING": config.TRAIN_CONVEX_HULL_SAMPLING,
    "VARIANCE_BOUND": config.TRAIN_VARIANCE_BOUND,
    "NR_QUERIES_PER_ITERATION": config.NR_QUERIES_PER_ITERATION,
    "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
    "CLASSIFIER": config.TRAIN_CLASSIFIER,
    "GENERATE_NOISE": config.TRAIN_GENERATE_NOISE,
    "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
    "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
    "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
    "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
    "STATE_LRU_AREAS_LIMIT": config.TRAIN_STATE_LRU_AREAS_LIMIT,
    "STATE_NO_LRU_WEIGHTS": config.TRAIN_STATE_NO_LRU_WEIGHTS,
}
param_string = train_base_param_string

OUTPUT_DIRECTORY = PARENT_OUTPUT_DIRECTORY + param_string

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

print("Saving to " + OUTPUT_DIRECTORY)

print("#" * 80)
print("Creating dataset")
print("#" * 80)
print("\n")

start = time.time()

if (
    not Path(OUTPUT_DIRECTORY + "/states.csv").is_file()
    or sum(1 for l in open(OUTPUT_DIRECTORY + "/states.csv"))
    < params["NR_LEARNING_SAMPLES"]
):
    if Path(OUTPUT_DIRECTORY + "/states.csv").is_file():
        NR_LEARNING_SAMPLES = params["NR_LEARNING_SAMPLES"] - sum(
            1 for l in open(OUTPUT_DIRECTORY + "/states.csv")
        )
    else:
        NR_LEARNING_SAMPLES = params["NR_LEARNING_SAMPLES"]

    def create_dataset_sample(RANDOM_SEED):
        cli_arguments = (
            "python imit_training.py "
            + " --DATASETS_PATH ../datasets"
            + " --OUTPUT_DIRECTORY "
            + OUTPUT_DIRECTORY
            + " --CLUSTER dummy "
            + " --NR_QUERIES_PER_ITERATION "
            + str(params["NR_QUERIES_PER_ITERATION"])
            + " --DATASET_NAME synthetic "
            + " --START_SET_SIZE 1 "
            + " --USER_QUERY_BUDGET_LIMIT "
            + str(params["USER_QUERY_BUDGET_LIMIT"])
            + " --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1"
            + " --AMOUNT_OF_PEAKED_OBJECTS 20 "
            + " --MAX_AMOUNT_OF_WS_PEAKS 0 "
            + " --AMOUNT_OF_LEARN_ITERATIONS 1 "
            + " --AMOUNT_OF_FEATURES "
            + str(params["AMOUNT_OF_FEATURES"])
            + " --VARIANCE_BOUND "
            + str(params["VARIANCE_BOUND"])
            + " --CLASSIFIER "
            + str(params["CLASSIFIER"])
            + " --STATE_LRU_AREAS_LIMIT "
            + str(params["STATE_LRU_AREAS_LIMIT"])
        )

        for k in [
            "VARIABLE_DATASET",
            "NEW_SYNTHETIC_PARAMS",
            "HYPERCUBE",
            "CONVEX_HULL_SAMPLING",
            "STOP_AFTER_MAXIMUM_ACCURACY_REACHED",
            "GENERATE_NOISE",
            "STATE_DISTANCES",
            "STATE_ARGTHIRD_PROBAS",
            "STATE_ARGSECOND_PROBAS",
            "STATE_DIFF_PROBAS",
            "STATE_NO_LRU_WEIGHTS",
        ]:
            if params[k]:
                cli_arguments += " --" + k + " "
        print(cli_arguments)

        os.system(cli_arguments)
        return RANDOM_SEED

    nr_parallel_processes = int(
        math.ceil(
            NR_LEARNING_SAMPLES
            / (params["USER_QUERY_BUDGET_LIMIT"] / params["NR_QUERIES_PER_ITERATION"])
        )
    )
    if nr_parallel_processes == 0:
        nr_parallel_processes = params["NR_LEARNING_SAMPLES"] + 1

    with Parallel(n_jobs=multiprocessing.cpu_count()) as parallel:
        output = parallel(
            delayed(create_dataset_sample)(k) for k in range(1, nr_parallel_processes)
        )
    #  print(output)

assert os.path.exists(OUTPUT_DIRECTORY + "/states.csv")

end = time.time()

start = time.time()


end = time.time()
print("Done in ", end - start, " s\n")
start = time.time()

print("#" * 80)
print("Training ANN")
print("#" * 80)
print("\n")

if not Path(OUTPUT_DIRECTORY + "/trained_ann.pickle").is_file():
    print("Training ann")
    os.system(
        "python train_lstm.py --DATA_PATH "
        + OUTPUT_DIRECTORY
        + " --STATE_ENCODING listwise --TARGET_ENCODING binary --SAVE_DESTINATION "
        + OUTPUT_DIRECTORY
        + "/trained_ann.pickle --REGULAR_DROPOUT_RATE 0.1 --OPTIMIZER RMSprop --NR_HIDDEN_NEURONS 80 --NR_HIDDEN_LAYERS 2 --LOSS CosineSimilarity --EPOCHS 1000 --BATCH_SIZE 32 --ACTIVATION elu --RANDOM_SEED 1"
    )


assert os.path.exists(OUTPUT_DIRECTORY + "/trained_ann.pickle")

end = time.time()
print("Done in ", end - start, " s\n")
start = time.time()


end = time.time()
print("Done in ", end - start, " s\n")
start = time.time()

print("#" * 80)
print("Creating evaluation ann data")
print("#" * 80)
print("\n")
params = {
    "USER_QUERY_BUDGET_LIMIT": config.USER_QUERY_BUDGET_LIMIT,
    "VARIABLE_DATASET": config.TEST_VARIABLE_DATASET,
    "comparisons": config.TEST_COMPARISONS,
    # ["random", "uncertainty_max_margin"],
    "NR_EVALUATIONS": config.TEST_NR_LEARNING_SAMPLES,
    "AMOUNT_OF_FEATURES": config.TEST_AMOUNT_OF_FEATURES,
    "HYPERCUBE": config.TEST_HYPERCUBE,
    "NEW_SYNTHETIC_PARAMS": config.TEST_NEW_SYNTHETIC_PARAMS,
    "CONVEX_HULL_SAMPLING": config.TEST_CONVEX_HULL_SAMPLING,
    "NR_QUERIES_PER_ITERATION": config.NR_QUERIES_PER_ITERATION,
    "CLASSIFIER": config.TEST_CLASSIFIER,
    "GENERATE_NOISE": config.TEST_GENERATE_NOISE,
    "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
    "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
    "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
    "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
    "STATE_LRU_AREAS_LIMIT": config.TRAIN_STATE_LRU_AREAS_LIMIT,
    "STATE_NO_LRU_WEIGHTS": config.TRAIN_STATE_NO_LRU_WEIGHTS,
}

trained_ann_csv_path = config.OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + ".csv"

if (
    not Path(trained_ann_csv_path).is_file()
    or sum(1 for l in open(trained_ann_csv_path)) < params["NR_EVALUATIONS"]
):

    def run_evaluation(RANDOM_SEED):
        RANDOM_SEED += 100000
        cli_arguments = (
            "python single_al_cycle.py --NN_BINARY "
            + OUTPUT_DIRECTORY
            + "/trained_ann.pickle --OUTPUT_DIRECTORY "
            + trained_ann_csv_path
            + " --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION "
            + str(params["NR_QUERIES_PER_ITERATION"])
            + " --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT "
            + str(params["USER_QUERY_BUDGET_LIMIT"])
            + " --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1"
            + " --AMOUNT_OF_FEATURES "
            + str(params["AMOUNT_OF_FEATURES"])
            + " --CLASSIFIER "
            + str(params["CLASSIFIER"])
            + " --STATE_LRU_AREAS_LIMIT "
            + str(params["STATE_LRU_AREAS_LIMIT"])
        )

        for k in [
            "VARIABLE_DATASET",
            "NEW_SYNTHETIC_PARAMS",
            "HYPERCUBE",
            "CONVEX_HULL_SAMPLING",
            "GENERATE_NOISE",
            "STATE_DISTANCES",
            "STATE_ARGTHIRD_PROBAS",
            "STATE_ARGSECOND_PROBAS",
            "STATE_DIFF_PROBAS",
            "STATE_NO_LRU_WEIGHTS",
        ]:
            if params[k]:
                cli_arguments += " --" + k + " "
        print(cli_arguments)
        #  exit(-1)
        os.system(cli_arguments)

        return RANDOM_SEED

    with Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading") as parallel:
        output = parallel(
            delayed(run_evaluation)(k) for k in range(1, params["NR_EVALUATIONS"] + 1)
        )
    #  print(output)
    assert os.path.exists(trained_ann_csv_path)

    # rename sampling column
    p = Path(trained_ann_csv_path)
    text = p.read_text()
    text = text.replace("trained_nn", OUTPUT_DIRECTORY)
    p.write_text(text)

assert os.path.exists(trained_ann_csv_path)
amount_of_lines = sum(1 for l in open(trained_ann_csv_path))
print("Evaluation trained_nn size: {}".format(amount_of_lines))

end = time.time()
print("Done in ", end - start, " s\n")

# remove STATE stuff

params = {
    "USER_QUERY_BUDGET_LIMIT": config.USER_QUERY_BUDGET_LIMIT,
    "VARIABLE_DATASET": config.TEST_VARIABLE_DATASET,
    "comparisons": config.TEST_COMPARISONS,
    # ["random", "uncertainty_max_margin"],
    "NR_EVALUATIONS": config.TEST_NR_LEARNING_SAMPLES,
    "AMOUNT_OF_FEATURES": config.TEST_AMOUNT_OF_FEATURES,
    "HYPERCUBE": config.TEST_HYPERCUBE,
    "NEW_SYNTHETIC_PARAMS": config.TEST_NEW_SYNTHETIC_PARAMS,
    "CONVEX_HULL_SAMPLING": config.TEST_CONVEX_HULL_SAMPLING,
    "NR_QUERIES_PER_ITERATION": config.NR_QUERIES_PER_ITERATION,
    "CLASSIFIER": config.TEST_CLASSIFIER,
    "GENERATE_NOISE": config.TEST_GENERATE_NOISE,
}

CLASSIC_PREFIX = test_base_param_string

#
#  for k, v in params.items():
#      if k in ["USER_QUERY_BUDGET_LIMIT", "NR_QUERIES_PER_ITERATION"]:
#          if parser.get_default(k) == v:
#              continue
#      elif parser.get_default("TEST_" + k) == v:
#          continue
#      #  k = "".join([x[0] for x in k.split("_")])
#
#      if k == "comparisons" or k == "FINAL_PICTURE":
#          continue
#      #  k = "".join([x[0] for x in k.split("_")])
#
#      if v == True:
#          CLASSIC_PREFIX += "_TEST" + k.lower() + "_T"
#      elif v == False:
#          CLASSIC_PREFIX += "_TEST" + k.lower() + "_F"
#      else:
#          CLASSIC_PREFIX += "_TEST" + str(v)
#


print("#" * 80)
print("Creating classic evaluation data")
print("#" * 80)
print("\n")

# check if the other evaluation csvs already exist
for comparison in params["comparisons"]:
    start = time.time()
    COMPARISON_PATH = (
        PARENT_OUTPUT_DIRECTORY + "classics/" + comparison + CLASSIC_PREFIX + ".csv"
    )

    Path(COMPARISON_PATH).parent.mkdir(parents=True, exist_ok=True)
    #  Path(COMPARISON_PATH).touch()
    print(COMPARISON_PATH)
    if (
        not Path(COMPARISON_PATH).is_file()
        or sum(1 for l in open(COMPARISON_PATH)) < params["NR_EVALUATIONS"]
    ):

        def run_classic_evaluation(RANDOM_SEED):
            RANDOM_SEED += 100000
            cli_arguments = (
                "python single_al_cycle.py --OUTPUT_DIRECTORY "
                + COMPARISON_PATH
                + " --SAMPLING "
                + comparison
                + " --CLUSTER dummy --NR_QUERIES_PER_ITERATION "
                + str(params["NR_QUERIES_PER_ITERATION"])
                + " --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT "
                + str(params["USER_QUERY_BUDGET_LIMIT"])
                + " --RANDOM_SEED "
                + str(RANDOM_SEED)
                + " --N_JOBS 1"
                + " --AMOUNT_OF_FEATURES "
                + str(params["AMOUNT_OF_FEATURES"])
                + " --CLASSIFIER "
                + str(params["CLASSIFIER"])
            )

            for k in [
                "VARIABLE_DATASET",
                "NEW_SYNTHETIC_PARAMS",
                "HYPERCUBE",
                "CONVEX_HULL_SAMPLING",
                "GENERATE_NOISE",
            ]:
                if params[k]:
                    cli_arguments += " --" + k + " "

            print(cli_arguments)
            os.system(cli_arguments)

            return RANDOM_SEED

        with Parallel(
            n_jobs=multiprocessing.cpu_count(), backend="threading"
        ) as parallel:
            output = parallel(
                delayed(run_classic_evaluation)(k)
                for k in range(1, params["NR_EVALUATIONS"] + 1)
            )
        #  print(output)
    assert os.path.exists(COMPARISON_PATH)
    amount_of_lines = sum(1 for l in open(COMPARISON_PATH))
    end = time.time()
    print("Done in ", end - start, " s\n")

start = time.time()
print("#" * 80)
print("Generating evaluation CSV")
print("#" * 80)
print("\n")

if config.FINAL_PICTURE == "":
    comparison_path = (
        PARENT_OUTPUT_DIRECTORY
        + param_string
        + CLASSIC_PREFIX
        + "_".join(params["comparisons"])
        + ".csv"
    )
else:
    comparison_path = config.FINAL_PICTURE
print(comparison_path)

if not Path(comparison_path).is_file():
    df = pd.read_csv(
        trained_ann_csv_path, index_col=None, nrows=1 + params["NR_EVALUATIONS"],
    )

    for comparison in params["comparisons"]:
        df2 = pd.read_csv(
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            + CLASSIC_PREFIX
            + ".csv",
            index_col=None,
            nrows=1 + params["NR_EVALUATIONS"],
        )
        df = pd.concat([df, df2])

    #  print(df)
    df.to_csv(comparison_path, index=False)

assert os.path.exists(comparison_path)
end = time.time()
print("Done in ", end - start, " s\n")

start = time.time()


df = pd.read_csv(comparison_path)
random_mean = df.loc[df["sampling"] == "random"]["acc_test_oracle"].mean()

mm_mean = df.loc[df["sampling"] == "uncertainty_max_margin"]["acc_test_oracle"].mean()
rest = df.loc[
    (df["sampling"] != "random") & (df["sampling"] != "uncertainty_max_margin")
]["acc_test_oracle"].mean()

print("{} {} {}".format(random_mean, rest, mm_mean))

print("#" * 80)
print("Evaluation plots")
print("#" * 80)
print("\n")

splitted_path = os.path.split(comparison_path)

os.system(
    "python compare_distributions.py --CSV_FILE "
    + comparison_path
    + "  --GROUP_COLUMNS sampling"
    + " --SAVE_FILE "
    + splitted_path[0]
    + "/"
    + str(mm_mean - rest)
    + splitted_path[1]
    + " --TITLE "
    + comparison_path
)

end = time.time()
print("Done in ", end - start, " s\n")
