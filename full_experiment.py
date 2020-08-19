import pandas as pd
import os
import subprocess
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed

PARENT_OUTPUT_DIRECTORY = "tmp/"

params = {
    "NR_EVALUATIONS": 100,
    "VARIABLE_DATASET": True,
    "NR_LEARNING_SAMPLES": 100,
    "REPRESENTATIVE_FEATURES": False,
    "AMOUNT_OF_FEATURES": -1,
    "HYPERCUBE": True,
    "OLD_SYNTHETIC_PARAMS": False,
}

param_string = ""

for k, v in params.items():
    if type(v) == bool:
        param_string += "_" + k.lower()
    else:
        param_string += "_" + str(v)

OUTPUT_DIRECTORY = PARENT_OUTPUT_DIRECTORY + param_string

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

print("Saving to " + OUTPUT_DIRECTORY)

print("#" * 80)
print("Creating dataset")
print("#" * 80)
print("\n")


if (
    not Path(OUTPUT_DIRECTORY + "/states.csv").is_file()
    or sum(1 for l in open(OUTPUT_DIRECTORY + "/states.csv"))
    < params["NR_LEARNING_SAMPLES"]
):

    def create_dataset_sample(RANDOM_SEED):
        cli_arguments = (
            "python imit_training.py "
            + " --DATASETS_PATH ../datasets"
            + " --OUTPUT_DIRECTORY "
            + OUTPUT_DIRECTORY
            + " --CLUSTER dummy "
            + " --NR_QUERIES_PER_ITERATION 5 "
            + " --DATASET_NAME synthetic "
            + " --START_SET_SIZE 1 "
            + " --USER_QUERY_BUDGET_LIMIT 50 "
            + " --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1"
            + " --AMOUNT_OF_PEAKED_OBJECTS 20 "
            + " --MAX_AMOUNT_OF_WS_PEAKS 0 "
            + " --AMOUNT_OF_LEARN_ITERATIONS 1 "
            + " --AMOUNT_OF_FEATURES "
            + str(params["AMOUNT_OF_FEATURES"])
        )

        if params["VARIABLE_DATASET"]:
            cli_arguments += " --VARIABLE_INPUT_SIZE "
        if params["REPRESENTATIVE_FEATURES"]:
            cli_arguments += " --REPRESENTATIVE_FEATURES "
        if params["OLD_SYNTHETIC_PARAMS"]:
            cli_arguments += " --OLD_SYNTHETIC_PARAMS "
        if params["HYPERCUBE"]:
            cli_arguments += " --HYPERCUBE "

        os.system(cli_arguments)
        return RANDOM_SEED

    nr_parallel_processes = int(params["NR_LEARNING_SAMPLES"] / 10)
    if nr_parallel_processes == 0:
        nr_parallel_processes = params["NR_LEARNING_SAMPLES"] + 1

    with Parallel(n_jobs=multiprocessing.cpu_count()) as parallel:
        output = parallel(
            delayed(create_dataset_sample)(k) for k in range(1, nr_parallel_processes)
        )
    print(output)

assert os.path.exists(OUTPUT_DIRECTORY + "/states.csv")

print("Created states:" + str(sum(1 for l in open(OUTPUT_DIRECTORY + "/states.csv"))))
print("Created opt_pol:" + str(sum(1 for l in open(OUTPUT_DIRECTORY + "/opt_pol.csv"))))

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
        + "/trained_ann.pickle --REGULAR_DROPOUT_RATE 0.1 --OPTIMIZER RMSprop --NR_HIDDEN_NEURONS 20 --NR_HIDDEN_LAYERS 2 --LOSS CosineSimilarity --EPOCHS 1000 --BATCH_SIZE 16 --ACTIVATION elu --RANDOM_SEED 1"
    )


assert os.path.exists(OUTPUT_DIRECTORY + "/trained_ann.pickle")

print("#" * 80)
print("Creating evaluation ann data")
print("#" * 80)
print("\n")

if (
    not Path(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv").is_file()
    or sum(1 for l in open(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv"))
    < params["NR_EVALUATIONS"]
):

    def run_evaluation(RANDOM_SEED):
        cli_arguments = (
            "python single_al_cycle.py --NN_BINARY "
            + OUTPUT_DIRECTORY
            + "/trained_ann.pickle --OUTPUT_DIRECTORY "
            + OUTPUT_DIRECTORY
            + "/trained_ann_evaluation.csv --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1"
            + " --AMOUNT_OF_FEATURES "
            + str(params["AMOUNT_OF_FEATURES"])
        )

        if params["VARIABLE_DATASET"]:
            cli_arguments += " --VARIABLE_INPUT_SIZE "
        if params["REPRESENTATIVE_FEATURES"]:
            cli_arguments += " --REPRESENTATIVE_FEATURES "
        if params["OLD_SYNTHETIC_PARAMS"]:
            cli_arguments += " --OLD_SYNTHETIC_PARAMS "
        if params["HYPERCUBE"]:
            cli_arguments += " --HYPERCUBE "

        os.system(cli_arguments)

        return RANDOM_SEED

    with Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading") as parallel:
        output = parallel(
            delayed(run_evaluation)(k) for k in range(1, params["NR_EVALUATIONS"] + 1)
        )
    print(output)

    # rename sampling column
    p = Path(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv")
    text = p.read_text()
    text = text.replace("trained_nn", OUTPUT_DIRECTORY)
    p.write_text(text)

assert os.path.exists(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv")

amount_of_lines = sum(1 for l in open(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv"))
print("Evaluation trained_nn size: {}".format(amount_of_lines))

print("#" * 80)
print("Creating classic evaluation data")
print("#" * 80)
print("\n")
params = {
    "VARIABLE_DATASET": True,
    "comparisons": ["uncertainty_max_margin"],  # ["random", "uncertainty_max_margin"],
    "NR_EVALUATIONS": 100,
    "REPRESENTATIVE_FEATURES": False,
    "AMOUNT_OF_FEATURES": -1,
    "HYPERCUBE": True,
    "OLD_SYNTHETIC_PARAMS": False,
}

CLASSIC_PREFIX = ""

for k, v in params.items():
    if type(v) == bool:
        CLASSIC_PREFIX += "_" + k.lower()
    else:
        CLASSIC_PREFIX += "_" + str(v)


# check if the other evaluation csvs already exist
for comparison in params["comparisons"]:
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
            cli_arguments = (
                "python single_al_cycle.py --OUTPUT_DIRECTORY "
                + COMPARISON_PATH
                + " --SAMPLING "
                + comparison
                + " --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED "
                + str(RANDOM_SEED)
                + " --N_JOBS 1"
                + " --AMOUNT_OF_FEATURES "
                + str(params["AMOUNT_OF_FEATURES"])
            )

            if params["VARIABLE_DATASET"]:
                cli_arguments += " --VARIABLE_INPUT_SIZE "
            if params["REPRESENTATIVE_FEATURES"]:
                cli_arguments += " --REPRESENTATIVE_FEATURES "
            if params["OLD_SYNTHETIC_PARAMS"]:
                cli_arguments += " --OLD_SYNTHETIC_PARAMS "
            if params["HYPERCUBE"]:
                cli_arguments += " --HYPERCUBE "

            os.system(cli_arguments)

            return RANDOM_SEED

        with Parallel(
            n_jobs=multiprocessing.cpu_count(), backend="threading"
        ) as parallel:
            output = parallel(
                delayed(run_classic_evaluation)(k)
                for k in range(1, params["NR_EVALUATIONS"] + 1)
            )
        print(output)
    assert os.path.exists(COMPARISON_PATH)
    amount_of_lines = sum(1 for l in open(COMPARISON_PATH))
    print("Evaluation " + comparison + "size: {}".format(amount_of_lines))


print("#" * 80)
print("Generating evaluation CSV")
print("#" * 80)
print("\n")

#  -> hier drinnen fehlt 1000_fixed und so :)
comparison_path = OUTPUT_DIRECTORY + "_".join(params["comparisons"]) + ".csv"
print(comparison_path)

if not Path(comparison_path).is_file():
    df = pd.read_csv(
        OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv",
        index_col=None,
        nrows=1 + params["NR_EVALUATIONS"],
    )

    for comparison in params["comparisons"]:
        df2 = pd.read_csv(
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            + "_"
            + CLASSIC_PREFIX
            + ".csv",
            index_col=None,
            nrows=1 + params["NR_EVALUATIONS"],
        )
        df = pd.concat([df, df2])

    print(df)
    df.to_csv(comparison_path, index=False)


assert os.path.exists(comparison_path)

print("#" * 80)
print("Evaluation plots")
print("#" * 80)
print("\n")

os.system(
    "python compare_distributions.py --CSV_FILE "
    + comparison_path
    + "  --GROUP_COLUMNS sampling"
    + " --SAVE_FILE "
    + comparison_path
)
