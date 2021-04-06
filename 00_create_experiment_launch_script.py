import argparse
import glob
import os
import stat
import sys

from jinja2 import Template

"""
1. 01_create_synthetic_training_data.py creates synthetic training data
(optional: 2. 02_hyper_search_or_train_imital.py hyper search)
3. 03_train_imital.py trains ann using default hyperparams or the ones from step 2

variant a: (old variant - deprecated (code is in archived))
4. classics.py  random + uncertainty baselines data
5. ann_eval_data.py eva data for the from step 3 trained ANN
6. plots.py creates some basic plots using the data from 4. and 5.

variant b: (new ALiPy variant)
4. 04_alipy_init_seeds.py creaets a CSV containing all the needed data for step 5
5. 05_alipy_eva.py actually is intended to run in a batch mode wit the provided data and csv file from step 4
6. 06_sync_and_run_experiment.sh -> updates taurus, starts experiment there --> only those, where the data is not present yet! should be able to detect if we are already at step 4 and that only some data has to be run again etc.
-> 06 downloaded zuerst von taurus die neuen results (backup von den alten vorher),  startet dann schritt 4, und pushed das zeugs dann hoch (rsync)!
"""


parser = argparse.ArgumentParser()
parser.add_argument("--TITLE")
parser.add_argument("--TEST_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--TRAIN_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--ITERATIONS_PER_BATCH", default=10, type=int)
parser.add_argument("--SLURM_FILE_PATH", default="_experiment_slurm_files")
parser.add_argument("--HPC_WS_DIR", default="/lustre/ssd/ws/s5968580-IL_TD2")
parser.add_argument(
    "--OUTPUT_DIR", default="/lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch"
)
parser.add_argument("--WITH_HYPER_SEARCH", action="store_true")
parser.add_argument("--WITH_CLASSICS", action="store_true")
parser.add_argument("--WITH_PLOTS", action="store_true")
parser.add_argument("--WITH_TUD_EVAL", action="store_true")
parser.add_argument("--WITH_ALIPY", action="store_true")
parser.add_argument("--INITIAL_BATCH_SAMPLING_ARG", type=int, default=200)
parser.add_argument("--INITIAL_BATCH_SAMPLING_HYBRID_UNCERT", type=float, default=0.2)
parser.add_argument("--INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST", type=float, default=0.2)
parser.add_argument(
    "--INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB", type=float, default=0.2
)
parser.add_argument(
    "--INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY", type=float, default=0.2
)
parser.add_argument("--TRAIN_STATE_DISTANCES", action="store_true")
parser.add_argument("--TRAIN_STATE_UNCERTAINTIES", action="store_true")
parser.add_argument("--TRAIN_STATE_PREDICTED_UNITY", action="store_true")
parser.add_argument("--TRAIN_STATE_DISTANCES_LAB", action="store_true")
parser.add_argument("--STATE_INCLUDE_NR_FEATURES", action="store_true")
parser.add_argument("--DISTANCE_METRIC", default="euclidean")
parser.add_argument("--TOTAL_BUDGET", type=int, default=50)
parser.add_argument("--EVA_DATASETS", nargs="*", default=["synthetic"])


# FIXME wenn HYBRID -> HYBRID namen so ändern, dass die Werte von oben an den titel angefügt werden

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


if config.WITH_CLASSICS or config.WITH_TUD_EVAL or config.WITH_PLOTS:
    print("config option deprecated")
    exit(-1)

INITIAL_BATCH_SAMPLING_METHOD = config.TITLE
if config.TITLE == "hybrid":
    config.TITLE = (
        "hybrid-"
        + str(config.INITIAL_BATCH_SAMPLING_ARG)
        + "#"
        + str(config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST)
        + "_"
        + str(config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB)
        + "_"
        + str(config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY)
        + "_"
        + str(config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT)
    )

if config.TITLE == "single":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    ADDITIONAL_TRAINING_STATE_ARGS = ""
elif config.TITLE == "single_full":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    ADDITIONAL_TRAINING_STATE_ARGS = " --STATE_ARGSECOND_PROBAS --STATE_ARGTHIRD_PROBAS --STATE_DISTANCES_LAB --STATE_DISTANCES_UNLAB "
    config.INITIAL_BATCH_SAMPLING_ARG = 10
elif config.TITLE == "single_full_f1":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    ADDITIONAL_TRAINING_STATE_ARGS = " --STATE_ARGSECOND_PROBAS --STATE_ARGTHIRD_PROBAS --STATE_DISTANCES_LAB --STATE_DISTANCES_UNLAB "
elif config.TITLE == "single_full_10":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    config.INITIAL_BATCH_SAMPLING_ARG = 10
    ADDITIONAL_TRAINING_STATE_ARGS = " --STATE_ARGSECOND_PROBAS --STATE_ARGTHIRD_PROBAS --STATE_DISTANCES_LAB --STATE_DISTANCES_UNLAB "
elif config.TITLE == "single_full_lab":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    ADDITIONAL_TRAINING_STATE_ARGS = " --STATE_ARGSECOND_PROBAS --STATE_DISTANCES_LAB "
elif config.TITLE == "single_full_unlab":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
    ADDITIONAL_TRAINING_STATE_ARGS = (
        " --STATE_ARGSECOND_PROBAS --STATE_DISTANCES_UNLAB "
    )
else:
    BATCH_MODE = "--BATCH_MODE"
    ADDITIONAL_TRAINING_STATE_ARGS = ""
    if config.TRAIN_STATE_DISTANCES:
        ADDITIONAL_TRAINING_STATE_ARGS += " --STATE_DISTANCES"
        config.TITLE += "_D"
    if config.TRAIN_STATE_UNCERTAINTIES:
        ADDITIONAL_TRAINING_STATE_ARGS += " --STATE_UNCERTAINTIES"
        config.TITLE += "_U"
    if config.TRAIN_STATE_PREDICTED_UNITY:
        ADDITIONAL_TRAINING_STATE_ARGS += " --STATE_PREDICTED_UNITY"
        config.TITLE += "_P"
    if config.TRAIN_STATE_DISTANCES_LAB:
        ADDITIONAL_TRAINING_STATE_ARGS += " --STATE_DISTANCES_LAB"
        config.TITLE += "_DL"

if config.STATE_INCLUDE_NR_FEATURES:
    ADDITIONAL_TRAINING_STATE_ARGS += " --STATE_INCLUDE_NR_FEATURES"

if config.DISTANCE_METRIC == "cosine":
    config.TITLE += "_cos"
if config.STATE_INCLUDE_NR_FEATURES:
    config.TITLE += "_nrf"

config.SLURM_FILE_PATH = config.SLURM_FILE_PATH + "/" + config.TITLE

slurm_common_template = Template(
    """#!/bin/bash{% if array %}{% set THREADS = 1 %}{% set MEMORY = 2583 %}{% endif %}
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{ THREADS }}
#SBATCH --mem-per-cpu={{ MEMORY }}M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output {{HPC_WS_DIR}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_WS_DIR}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_error.txt
{% if array %}#SBATCH --array {{START}}-{{END}}{% endif %}

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

{% if array %}i=$(( {{OFFSET}} + $SLURM_ARRAY_TASK_ID * {{ITERATIONS_PER_BATCH}} )){% endif %}

MPLCONFIGDIR={{HPC_WS_DIR}}/cache python3 -m pipenv run python {{HPC_WS_DIR}}/imitating-weakal/{{PYTHON_FILE}}.py {{ CLI_ARGS }}
exit 0
"""
)

bash_mode_common_template = Template("{{PYTHON_FILE}}.py {{ CLI_ARGS }}")

submit_jobs = Template(
    """#!/bin/bash
01_create_synthetic_training_data_id=$(sbatch --parsable {{HPC_WS_DIR}}/imitating-weakal/{{SLURM_FILE_PATH}}/01_create_synthetic_training_data.slurm)
{%if WITH_HYPER_SEARCH %}hyper_search_id=$(sbatch --parsable --dependency=afterok:$01_create_synthetic_training_data_id {{HPC_WS_DIR}}/imitating-weakal/{{SLURM_FILE_PATH}}/02_hyper_search.slurm){% endif %}
03_train_imital_id=$(sbatch --parsable --dependency=afterok:$01_create_synthetic_training_data_id{%if WITH_HYPER_SEARCH %}:$hyper_search_id{% endif %} {{HPC_WS_DIR}}/imitating-weakal/{{SLURM_FILE_PATH}}/03_train_imital.slurm)
{%if WITH_TUD_EVAL %}create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$01_create_synthetic_training_data_id:$03_train_imital_id{%if WITH_HYPER_SEARCH %}:$hyper_search_id{% endif %} {{HPC_WS_DIR}}/imitating-weakal//{{SLURM_FILE_PATH}}/ann_eval_data.slurm){% endif %}
{%if WITH_CLASSICS %}classics_id=$(sbatch --parsable {{HPC_WS_DIR}}/imitating-weakal//{{SLURM_FILE_PATH}}/classics.slurm){% endif %}
{%if WITH_PLOTS %}plots_id=$(sbatch --parsable --dependency=afterok:$01_create_synthetic_training_data_id{%if WITH_TUD_EVAL %}:$create_ann_eval_id{% endif %}{%if WITH_CLASSICS %}:$classics_id{% endif %} {{HPC_WS_DIR}}/imitating-weakal//{{SLURM_FILE_PATH}}/plots.slurm){% endif %}
exit 0
"""
)


sync_to_taurus = Template(
    """
    """
)


def write_slurm_and_bash_file(OUTPUT_FILE: str, **kwargs):
    with open(config.SLURM_FILE_PATH + "/" + OUTPUT_FILE + ".slurm", "w") as f:
        f.write(slurm_common_template.render(**kwargs))
    with open(config.SLURM_FILE_PATH + "/" + OUTPUT_FILE + ".tmp", "w") as f:
        f.write(bash_mode_common_template.render(**kwargs))


if not os.path.exists(config.SLURM_FILE_PATH):
    os.makedirs(config.SLURM_FILE_PATH)


START = 0
END = int(config.TRAIN_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1
write_slurm_and_bash_file(
    OUTPUT_FILE="01_create_synthetic_training_data",
    HPC_WS_DIR=config.HPC_WS_DIR,
    TITLE=config.TITLE,
    PYTHON_FILE="01_create_synthetic_training_data",
    array=True,
    START=START,
    END=END,
    ITERATIONS_PER_BATCH=config.ITERATIONS_PER_BATCH,
    OFFSET=0,
    CLI_ARGS=" "
    + str(BATCH_MODE)
    + " --INITIAL_BATCH_SAMPLING_METHOD "
    + str(INITIAL_BATCH_SAMPLING_METHOD)
    + " --BASE_PARAM_STRING batch_"
    + config.TITLE
    + " --INITIAL_BATCH_SAMPLING_ARG "
    + str(config.INITIAL_BATCH_SAMPLING_ARG)
    + " --OUTPUT_DIRECTORY "
    + config.OUTPUT_DIR
    + " --TOTAL_BUDGET "
    + str(config.TOTAL_BUDGET)
    + " --NR_LEARNING_SAMPLES "
    + str(config.ITERATIONS_PER_BATCH)
    + " --INITIAL_BATCH_SAMPLING_HYBRID_UNCERT "
    + str(config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT)
    + " --INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY "
    + str(config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY)
    + " --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST "
    + str(config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST)
    + " --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB "
    + str(config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB)
    + ADDITIONAL_TRAINING_STATE_ARGS
    + " --RANDOM_ID_OFFSET $i"
    + " --DISTANCE_METRIC "
    + str(config.DISTANCE_METRIC),
)


if config.WITH_HYPER_SEARCH:
    write_slurm_and_bash_file(
        OUTPUT_FILE="02_hyper_search",
        HPC_WS_DIR=config.HPC_WS_DIR,
        TITLE=config.TITLE,
        PYTHON_FILE="02_hyper_search_or_train_imital",
        array=False,
        THREADS=24,
        MEMORY=5250,
        CLI_ARGS="--DATA_PATH "
        + config.OUTPUT_DIR
        + "/batch_"
        + config.TITLE
        + " --STATE_ENCODING listwise --TARGET_ENCODING binary --HYPER_SEARCH --N_ITER 100 ",
    )

if config.WITH_HYPER_SEARCH:
    hypered_appendix = " --HYPER_SEARCHED"
else:
    hypered_appendix = ""
write_slurm_and_bash_file(
    OUTPUT_FILE="03_train_imital",
    HPC_WS_DIR=config.HPC_WS_DIR,
    TITLE=config.TITLE,
    PYTHON_FILE="03_train_imital",
    array=False,
    THREADS=8,
    MEMORY=5250,
    CLI_ARGS="--OUTPUT_DIRECTORY "
    + config.OUTPUT_DIR
    + "/ --BASE_PARAM_STRING batch_"
    + config.TITLE
    + hypered_appendix,
)


if config.WITH_ALIPY:
    alipy_init_seeds_template = Template(
        """#!/bin/bash
    python ../04_alipy_init_seeds.py --OUTPUT_PATH {{ OUTPUT_PATH }} --DATASETS {{ DATASETS }} --AMOUNT_OF_RUNS {{ AMOUNT_OF_EVAL_RUNS }} --NON_SLURM --SLURM_FILE_TO_UPDATE {{ SLURM_FILE_TO_UPDATE }}
    python ../04_alipy_init_seeds.py --OUTPUT_PATH {{ OUTPUT_PATH }} --DATASETS {{ DATASETS }} --AMOUNT_OF_RUNS {{ AMOUNT_OF_EVAL_RUNS }} --SLURM_FILE_TO_UPDATE {{ SLURM_FILE_TO_UPDATE }}
    """
    )

    # if this file is run, it automatically updates the ARRAY indices for the alipy slurm job based on the result of this python script
    with open(config.SLURM_FILE_PATH + "/04_alipy_init_seeds.sh", "w") as f:
        START = 0
        END = int(config.TEST_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1
        f.write(
            alipy_init_seeds_template.render(
                OUTPUT_PATH=config.OUTPUT_DIR,
                DATASETS=",".join(config.EVA_DATASETS),
                AMOUNT_OF_RUNS=config.TEST_NR_LEARNING_SAMPLES,
                SLURM_FILE_TO_UPDATE=config.SLURM_FILE_PATH + "/05_alipy_eva.slurm",
            )
        )

    with open(config.SLURM_FILE_PATH + "/05_alipy_eva.slurm", "w") as f:
        START = 0
        END = int(config.TEST_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1
        f.write(
            slurm_common_template.render(
                HPC_WS_DIR=config.HPC_WS_DIR,
                TITLE=config.TITLE,
                PYTHON_FILE="04_alipy_init_seeds.py",
                array=True,
                START=START,
                END=END,
                THREADS=2,
                MEMORY=2583,
                CLI_ARGS=" "
                + " --OUTPUT_PATH "
                + config.OUTPUT_DIR
                + "/ --INDEX $SLURM_ARRAY_TASK_ID",
            )
        )


with open(config.SLURM_FILE_PATH + "/submit_slurm_jobs.sh", "w") as f:
    f.write(
        submit_jobs.render(
            HPC_WS_DIR=config.HPC_WS_DIR,
            SLURM_FILE_PATH=config.SLURM_FILE_PATH,
            TITLE=config.TITLE,
            WITH_HYPER_SEARCH=config.WITH_HYPER_SEARCH,
            WITH_ALIPY=config.WITH_ALIPY,
        )
    )

# open all fake slurms and concat them into a single bash file
submit_content = "#!/bin/bash\n"
sort_order = {
    "01_create_synthetic_training_data.tmp": 0,
    "02_hyper_search.tmp": 1,
    "03_train_imital.tmp": 2,
    "04_alipy_init_seeds.tmp": 3,
    "05_alipy_eva.tmp": 4,
}
for tmp_file in sorted(
    list(glob.glob(str(config.SLURM_FILE_PATH) + "/*.tmp")),
    key=lambda v: sort_order[v.split("/")[-1]],
):
    with open(tmp_file, "r") as f:
        content = f.read()
        content = content.replace("$i", "0")
        submit_content += "python " + content + "\n"
    os.remove(tmp_file)
with open(config.SLURM_FILE_PATH + "/submit_bash_jobs.sh", "w") as f:
    f.write(submit_content)
st = os.stat(config.SLURM_FILE_PATH + "/submit_bash_jobs.sh")
os.chmod(config.SLURM_FILE_PATH + "/submit_bash_jobs.sh", st.st_mode | stat.S_IEXEC)
