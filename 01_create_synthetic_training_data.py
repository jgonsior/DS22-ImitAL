from imitLearningPipelineSharedCode import get_config, run_parallel_experiment

(
    config,
    shared_arguments,
    _,
    ann_arguments,
    PARENT_OUTPUT_DIRECTORY,
    base_param_string,
) = get_config()


OUTPUT_FILE = PARENT_OUTPUT_DIRECTORY + base_param_string
run_parallel_experiment(
    "Creating dataset",
    OUTPUT_FILE=PARENT_OUTPUT_DIRECTORY
    + base_param_string
    + "/01_dataset_creation_stats.csv",
    CLI_COMMAND="python synthetic_datasets_imitation_training.py",
    CLI_ARGUMENTS={
        "DATASETS_PATH": "../datasets",
        "CLASSIFIER": config.CLASSIFIER,
        "OUTPUT_DIRECTORY": PARENT_OUTPUT_DIRECTORY + base_param_string,
        "DATASET_NAME": "synthetic",
        "SAMPLING": "trained_nn",
        "AMOUNT_OF_PEAKED_OBJECTS": config.AMOUNT_OF_PEAKED_SAMPLES,
        "AMOUNT_OF_FEATURES": config.AMOUNT_OF_FEATURES,
        "VARIABLE_DATASET": config.VARIABLE_DATASET,
        "NEW_SYNTHETIC_PARAMS": config.NEW_SYNTHETIC_PARAMS,
        "HYPERCUBE": config.HYPERCUBE,
        "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
        "GENERATE_NOISE": config.GENERATE_NOISE,
        **ann_arguments,
        **shared_arguments,
    },
    RANDOM_ID_OFFSET=config.RANDOM_ID_OFFSET,
    PARALLEL_AMOUNT=config.NR_LEARNING_SAMPLES,
    OUTPUT_FILE_LENGTH=config.NR_LEARNING_SAMPLES,
    RESTART_IF_NOT_ENOUGH_SAMPLES=False,
)
