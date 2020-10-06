import pandas as pd
import os
import subprocess
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
from active_learning.experiment_setup_lib import standard_config


smallest_margin_mm_rest = 100
smallest_margin_mm_rest_file = ""
biggest_margin_random_rest = 0
biggest_margin_random_rest_file = ""

for csv_file in Path(
    "tmp_100/plots_TRAIN_VARIABLE_DATASET_TRAIN_REPRESENTATIVE_FEATURES_TRAIN_AMOUNT_OF_FEATURES_TRAIN_NEW_SYNTHETIC_PARAMS_TEST_VARIABLE_DATASET_TEST_REPRESENTATIVE_FEATURES_TEST_AMOUNT_OF_FEATURES_TEST_NEW_SYNTHETIC_PARAMS"
).rglob("*"):
    csv_path_name = str(csv_file)
    if csv_path_name.endswith(".pdf") or csv_path_name.endswith(".png"):
        continue

    #  print(csv_path_name)

    df = pd.read_csv(csv_path_name)
    random_mean = df.loc[df["sampling"] == "random"]["acc_test_oracle"].mean()

    mm_mean = df.loc[df["sampling"] == "uncertainty_max_margin"][
        "acc_test_oracle"
    ].mean()
    rest = df.loc[
        (df["sampling"] != "random") & (df["sampling"] != "uncertainty_max_margin")
    ]["acc_test_oracle"].mean()

    if mm_mean - rest < smallest_margin_mm_rest:
        smallest_margin_mm_rest = mm_mean - rest
        smallest_margin_mm_rest_file = csv_path_name
    if rest - random_mean > biggest_margin_random_rest:
        biggest_margin_random_rest = rest - random_mean
        biggest_margin_random_rest_file = csv_path_name
    #  print("{} {} {}".format(random_mean, rest, mm_mean))

print(smallest_margin_mm_rest)
print(smallest_margin_mm_rest_file)

print(biggest_margin_random_rest)
print(biggest_margin_random_rest_file)
exit(-1)


amount_of_lines = sum(1 for l in open(trained_ann_csv_path))
print("Evaluation trained_nn size: {}".format(amount_of_lines))

print("#" * 80)
print("Creating classic evaluation data")
print("#" * 80)
print("\n")

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
            if params["NEW_SYNTHETIC_PARAMS"]:
                cli_arguments += " --NEW_SYNTHETIC_PARAMS "
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
        #  print(output)
    assert os.path.exists(COMPARISON_PATH)
    amount_of_lines = sum(1 for l in open(COMPARISON_PATH))
    print("Evaluation " + comparison + "size: {}".format(amount_of_lines))


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
