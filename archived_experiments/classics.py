from imitLearningPipelineSharedCode import get_config, run_parallel_experiment

(
    config,
    shared_arguments,
    evaluation_arguments,
    _,
    PARENT_OUTPUT_PATH,
    base_param_string,
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
    original_base_param_string = base_param_string
    base_param_string += "_" + DATASET_NAME
    if DATASET_NAME != "synthetic":
        #  config.NR_LEARNING_SAMPLES = 100
        evaluation_arguments["TOTAL_BUDGET"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    for comparison in config.TEST_COMPARISONS:
        #  print(comparison)
        COMPARISON_PATH = (
            PARENT_OUTPUT_PATH
            + "classics/"
            + comparison
            #  + base_param_string
            + ".csv"
        )
        run_parallel_experiment(
            "Creating " + comparison + "-evaluation data",
            OUTPUT_FILE=COMPARISON_PATH,
            CLI_COMMAND="python single_al_cycle.py",
            CLI_ARGUMENTS={
                "OUTPUT_PATH": COMPARISON_PATH,
                "QUERY_STRATEGY": comparison,
                **evaluation_arguments,
            },
            RANDOM_ID_OFFSET=config.RANDOM_ID_OFFSET,
            PARALLEL_AMOUNT=config.NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.NR_LEARNING_SAMPLES,
        )
    base_param_string = original_base_param_string
