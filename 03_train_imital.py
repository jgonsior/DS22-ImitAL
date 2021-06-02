import json
import os

from imitLearningPipelineSharedCode import get_config, run_python_experiment

(
    config,
    _,
    _,
    _,
    PARENT_OUTPUT_PATH,
    base_param_string,
) = get_config()

print(base_param_string)

if config.HYPER_SEARCHED:
    HYPER_SEARCH_OUTPUT_FILE = (
        config.OUTPUT_PATH + base_param_string + "/hyper_results.txt"
    )

    with open(HYPER_SEARCH_OUTPUT_FILE, "r") as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        lower_params = json.loads(last_line)
        ANN_HYPER_PARAMS = {}
        for k, v in lower_params.items():
            ANN_HYPER_PARAMS[k.upper()] = v
else:
    if config.BATCH_MODE:
        ANN_HYPER_PARAMS = {
            "REGULAR_DROPOUT_RATE": 0.2,
            "OPTIMIZER": "RMSprop",
            "NR_HIDDEN_NEURONS": 900,
            "NR_HIDDEN_LAYERS": 3,
            "LOSS": "MeanSquaredError",
            "EPOCHS": 10000,
            "ANN_BATCH_SIZE": 128,
            "ACTIVATION": "elu",
        }
    else:
        ANN_HYPER_PARAMS = {
            "REGULAR_DROPOUT_RATE": 0.2,
            "OPTIMIZER": "Nadam",
            "NR_HIDDEN_NEURONS": 1100,
            "NR_HIDDEN_LAYERS": 2,
            "LOSS": "MeanSquaredError",
            "EPOCHS": 10000,
            "ANN_BATCH_SIZE": 128,
            "ACTIVATION": "elu",
        }


# prevent retraining!
if os.path.isfile(
    PARENT_OUTPUT_PATH
    + base_param_string
    + "/03_imital_trained_ann.model/saved_model.pb"
):
    exit(0)
    # pass


run_python_experiment(
    "Train ANN",
    PARENT_OUTPUT_PATH + base_param_string + "/03_imital_trained_ann.model",
    CLI_COMMAND="python 02_hyper_search_or_train_imital.py",
    CLI_ARGUMENTS={
        "OUTPUT_PATH": config.OUTPUT_PATH,
        "BASE_PARAM_STRING": base_param_string,
        "STATE_ENCODING": config.STATE_ENCODING,
        "TARGET_ENCODING": config.TARGET_ENCODING,
        "SAVE_DESTINATION": config.OUTPUT_PATH
        + "/"
        + base_param_string
        + "/03_imital_trained_ann.model",
        "REGULAR_DROPOUT_RATE": ANN_HYPER_PARAMS["REGULAR_DROPOUT_RATE"],
        "OPTIMIZER": ANN_HYPER_PARAMS["OPTIMIZER"],
        "NR_HIDDEN_NEURONS": ANN_HYPER_PARAMS["NR_HIDDEN_NEURONS"],
        "NR_HIDDEN_LAYERS": ANN_HYPER_PARAMS["NR_HIDDEN_LAYERS"],
        "LOSS": ANN_HYPER_PARAMS["LOSS"],
        "EPOCHS": ANN_HYPER_PARAMS["EPOCHS"],
        "ANN_BATCH_SIZE": ANN_HYPER_PARAMS["ANN_BATCH_SIZE"],
        "ACTIVATION": ANN_HYPER_PARAMS["ACTIVATION"],
        "RANDOM_SEED": 1,
        "PERMUTATE_NN_TRAINING_INPUT": config.PERMUTATE_NN_TRAINING_INPUT,
    },
)
