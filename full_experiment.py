import pandas as pd
import os
import subprocess
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed

variable_dataset_size = True
comparisons = ["random", "uncertainty_max_margin"]
NR_LEARNING_SAMPLES = 100
NR_EVALUATIONS = 100
PARENT_OUTPUT_DIRECTORY = "tmp/"
OUTPUT_DIRECTORY = "tmp/" + str(NR_LEARNING_SAMPLES) + "_"

if variable_dataset_size:
    VARIABLE_APPENDIX = "variable"
else:
    VARIABLE_APPENDIX = "fixed"

OUTPUT_DIRECTORY += VARIABLE_APPENDIX

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

print("Saving to " + OUTPUT_DIRECTORY)

print("Creating dataset")


if (
    not Path(OUTPUT_DIRECTORY + "/states.csv").is_file()
    or sum(1 for l in open(OUTPUT_DIRECTORY + "/states.csv")) < NR_LEARNING_SAMPLES
):

    def create_dataset_sample(RANDOM_SEED):
        os.system(
            "python imit_training.py --DATASETS_PATH ../datasets --OUTPUT_DIRECTORY "
            + OUTPUT_DIRECTORY
            + " --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --AMOUNT_OF_LEARN_ITERATIONS 1"
        )
        return RANDOM_SEED

    with Parallel(n_jobs=multiprocessing.cpu_count()) as parallel:
        output = parallel(
            delayed(create_dataset_sample)(k) for k in range(1, NR_LEARNING_SAMPLES + 1)
        )
    print(output)

print("Created states:" + str(sum(1 for l in open(OUTPUT_DIRECTORY + "/states.csv"))))
print("Created opt_pol:" + str(sum(1 for l in open(OUTPUT_DIRECTORY + "/opt_pol.csv"))))


if not Path(OUTPUT_DIRECTORY + "/trained_ann.pickle").is_file():
    print("Training ann")
    os.system(
        "python train_lstm.py --DATA_PATH "
        + OUTPUT_DIRECTORY
        + " --STATE_ENCODING listwise --TARGET_ENCODING binary --SAVE_DESTINATION "
        + OUTPUT_DIRECTORY
        + "/trained_ann.pickle --REGULAR_DROPOUT_RATE 0.1 --OPTIMIZER RMSprop --NR_HIDDEN_NEURONS 20 --NR_HIDDEN_LAYERS 2 --LOSS CosineSimilarity --EPOCHS 1000 --BATCH_SIZE 16 --ACTIVATION elu --RANDOM_SEED 1"
    )


if (
    not Path(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv").is_file()
    or sum(1 for l in open(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv"))
    < NR_EVALUATIONS
):

    def run_evaluation(RANDOM_SEED):
        os.system(
            "python single_al_cycle.py --NN_BINARY "
            + OUTPUT_DIRECTORY
            + "/trained_ann.pickle --OUTPUT_DIRECTORY "
            + OUTPUT_DIRECTORY
            + "/trained_ann_evaluation.csv --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED "
            + str(RANDOM_SEED)
            + " --N_JOBS 1"
        )
        return RANDOM_SEED

    with Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading") as parallel:
        output = parallel(
            delayed(run_evaluation)(k) for k in range(1, NR_EVALUATIONS + 1)
        )
    print(output)

    # rename sampling column
    p = Path(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv")
    text = p.read_text()
    text = text.replace("trained_nn", OUTPUT_DIRECTORY)
    p.write_text(text)

amount_of_lines = sum(1 for l in open(OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv"))
print("Evaluation trained_nn size: {}".format(amount_of_lines))


# check if the other evaluation csvs already exist
for comparison in comparisons:
    COMPARISON_PATH = (
        PARENT_OUTPUT_DIRECTORY
        + "classics/"
        + comparison
        + "_"
        + VARIABLE_APPENDIX
        + "_"
        + str(NR_EVALUATIONS)
        + ".csv"
    )

    Path(COMPARISON_PATH).parent.mkdir(parents=True, exist_ok=True)
    #  Path(COMPARISON_PATH).touch()
    print(COMPARISON_PATH)
    if (
        not Path(COMPARISON_PATH).is_file()
        or sum(1 for l in open(COMPARISON_PATH)) < NR_EVALUATIONS
    ):

        def run_classic_evaluation(RANDOM_SEED):
            os.system(
                "python single_al_cycle.py --OUTPUT_DIRECTORY "
                + COMPARISON_PATH
                + " --SAMPLING "
                + comparison
                + " --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED "
                + str(RANDOM_SEED)
                + " --N_JOBS 1"
            )
            return RANDOM_SEED

        with Parallel(
            n_jobs=multiprocessing.cpu_count(), backend="threading"
        ) as parallel:
            output = parallel(
                delayed(run_classic_evaluation)(k) for k in range(1, NR_EVALUATIONS + 1)
            )
        print(output)
    amount_of_lines = sum(1 for l in open(COMPARISON_PATH))
    print("Evaluation " + comparison + "size: {}".format(amount_of_lines))


comparison_path = (
    PARENT_OUTPUT_DIRECTORY
    + "comparison_"
    + str(NR_EVALUATIONS)
    + "_"
    + "_".join(comparisons)
    + ".csv"
)
print(comparison_path)

if not Path(comparison_path).is_file():
    df = pd.read_csv(
        OUTPUT_DIRECTORY + "/trained_ann_evaluation.csv",
        index_col=None,
        nrows=1 + NR_EVALUATIONS,
    )

    for comparison in comparisons:
        df2 = pd.read_csv(
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            + "_"
            + VARIABLE_APPENDIX
            + "_"
            + str(NR_EVALUATIONS)
            + ".csv",
            index_col=None,
            nrows=1 + NR_EVALUATIONS,
        )
        df = pd.concat([df, df2])

    print(df)
    df.to_csv(comparison_path, index=False)

os.system(
    "python compare_distributions.py --CSV_FILE "
    + comparison_path
    + "  --GROUP_COLUMNS sampling"
)
