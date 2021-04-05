from imitLearningPipelineSharedCode import get_config, run_parallel_experiment

(
    config,
    shared_arguments,
    evaluation_arguments,
    ann_arguments,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
) = get_config()

# FIXME: currently the same training and test params are being used here, but that depends on how the new evaluation will look like actually
# probably better to replace this here with alipy altogether?
test_base_param_string = train_base_param_string

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
        evaluation_arguments["TOTAL_BUDGET"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    EVALUATION_FILE_TRAINED_NN_PATH = (
        config.OUTPUT_DIRECTORY
        + "/"
        + config.BASE_PARAM_STRING
        + "_"
        + DATASET_NAME
        + ".csv"
    )

    run_parallel_experiment(
        "Creating ann-evaluation data",
        OUTPUT_FILE=EVALUATION_FILE_TRAINED_NN_PATH,
        CLI_COMMAND="python single_al_cycle.py",
        CLI_ARGUMENTS={
            "NN_BINARY": config.OUTPUT_DIRECTORY
            + train_base_param_string
            + "/trained_ann.pickle",
            "OUTPUT_DIRECTORY": EVALUATION_FILE_TRAINED_NN_PATH,
            "SAMPLING": "trained_nn",
            **ann_arguments,
            **evaluation_arguments,
        },
        RANDOM_ID_OFFSET=config.TEST_RANDOM_ID_OFFSET,
        PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
    )

    test_base_param_string = original_test_base_param_string
