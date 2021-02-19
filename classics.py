from experiments_lib import get_config, run_parallel_experiment

(
    config,
    shared_arguments,
    evaluation_arguments,
    _,
    PARENT_OUTPUT_DIRECTORY,
    _,
    test_base_param_string,
) = get_config()


for DATASET_NAME in [
    #  "emnist-byclass-test",
    "synthetic",
    #  "dwtc",
    #  "BREAST",
    #  "DIABETES",
    #  "FERTILITY",
    #  "GERMAN",
    #  "HABERMAN",
    #  "HEART",
    #  "ILPD",
    #  "IONOSPHERE",
    #  "PIMA",
    #  "PLANNING",
    #  "australian",
]:
    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME
    if DATASET_NAME != "synthetic":
        #  config.TEST_NR_LEARNING_SAMPLES = 100
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    for comparison in config.TEST_COMPARISONS:
        #  print(comparison)
        COMPARISON_PATH = (
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            #  + test_base_param_string
            + ".csv"
        )
        run_parallel_experiment(
            "Creating " + comparison + "-evaluation data",
            OUTPUT_FILE=COMPARISON_PATH,
            CLI_COMMAND="python single_al_cycle.py",
            CLI_ARGUMENTS={
                "OUTPUT_DIRECTORY": COMPARISON_PATH,
                "SAMPLING": comparison,
                **evaluation_arguments,
            },
            RANDOM_ID_OFFSET=config.TEST_RANDOM_ID_OFFSET,
            PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
        )
    test_base_param_string = original_test_base_param_string
