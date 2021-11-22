from imitLearningPipelineSharedCode import get_config, run_parallel_experiment

(
    config,
    shared_arguments,
    _,
    ann_arguments,
    PARENT_OUTPUT_PATH,
) = get_config()


OUTPUT_FILE = PARENT_OUTPUT_PATH
run_parallel_experiment(
    "Creating dataset",
    OUTPUT_FILE=PARENT_OUTPUT_PATH + "/01_dataset_creation_stats.csv",
    CLI_COMMAND="python synthetic_datasets_imitation_training.py",
    CLI_ARGUMENTS={
        "DATASETS_PATH": "../datasets",
        "CLASSIFIER": config.CLASSIFIER,
        "OUTPUT_PATH": PARENT_OUTPUT_PATH,
        "DATASET_NAME": "synthetic",
        "QUERY_STRATEGY": config.QUERY_STRATEGY,
        "AMOUNT_OF_PEAKED_OBJECTS": config.AMOUNT_OF_PEAKED_SAMPLES,
        "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
        "ANDREAS": config.ANDREAS,
        "ANDREAS_NUMBER": config.ANDREAS_NUMBER,
        **ann_arguments,
        **shared_arguments,
    },
    RANDOM_ID_OFFSET=config.RANDOM_ID_OFFSET,
    PARALLEL_AMOUNT=config.NR_LEARNING_SAMPLES,
    OUTPUT_FILE_LENGTH=config.NR_LEARNING_SAMPLES,
    RESTART_IF_NOT_ENOUGH_SAMPLES=False,
)
