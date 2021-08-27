import argparse
from configparser import RawConfigParser
import glob
import os
import stat
import sys
from jinja2 import Template

"""
1. 01_create_synthetic_training_data.py creates synthetic training data
(optional: 2. 02_hyper_search_or_train_imital.py hyper search)
3. 03_train_imital.py trains ann using default hyperparams or the ones from step 2
4. 04_alipy_init_seeds.py creaets a CSV containing all the needed data for step 5
5. 05_alipy_eva.py actually is intended to run in a batch mode wit the provided data and csv file from step 4
6. 06_sync_and_run_experiment.sh -> updates taurus, starts experiment there --> only those, where the data is not present yet! should be able to detect if we are already at step 4 and that only some data has to be run again etc.
-> 06 downloaded zuerst von taurus die neuen results (backup von den alten vorher),  startet dann schritt 4, und pushed das zeugs dann hoch (rsync)!
"""

# we have config from a config file AND CLI arguments -> they'll get joined later
config_parser = RawConfigParser()
config_parser.read(".server_access_credentials.cfg")

parser = argparse.ArgumentParser()
parser.add_argument("--EXP_TITLE")
parser.add_argument("--TEST_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--TRAIN_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--ITERATIONS_PER_BATCH", default=10, type=int)
parser.add_argument("--N_JOBS", default=4, type=int)
parser.add_argument("--EXPERIMENT_LAUNCH_SCRIPTS", default="_experiment_launch_scripts")
parser.add_argument("--WITH_HYPER_SEARCH", action="store_true")
parser.add_argument("--WITH_CLASSICS", action="store_true")
parser.add_argument("--WITH_PLOTS", action="store_true")
parser.add_argument("--WITH_TUD_EVAL", action="store_true")
parser.add_argument("--WITH_ALIPY", action="store_true")
parser.add_argument("--ONLY_ALIPY", action="store_true")
parser.add_argument("--PRE_SAMPLING_HYBRID_UNCERT", type=float, default=0.2)
parser.add_argument("--PRE_SAMPLING_HYBRID_FURTHEST", type=float, default=0.2)
parser.add_argument("--PRE_SAMPLING_HYBRID_FURTHEST_LAB", type=float, default=0.2)
parser.add_argument("--PRE_SAMPLING_HYBRID_PRED_UNITY", type=float, default=0.2)

parser.add_argument("--STATE_INCLUDE_NR_FEATURES", action="store_true")
parser.add_argument("--TOTAL_BUDGET", type=int, default=50)

parser.add_argument("--WS_MODE", action="store_true")
parser.add_argument("--USE_WS_LABELS_CONTINOUSLY", action="store_true")

parser.add_argument("--EVA_DATASET_IDS", nargs="*", default=[0])
parser.add_argument("--EVA_STRATEGY_IDS", nargs="*", default=[0, 1, 2, 12])
parser.add_argument("--PERMUTATE_NN_TRAINING_INPUT", type=int, default=0)
parser.add_argument("--TARGET_ENCODING", default="binary")
parser.add_argument("--ANDREAS", default="None")

# parser.add_argument("--BATCH_MODE", action="store_true")
parser.add_argument(
    "--STATE_ARGS",
    nargs="*",
    default=[
        "STATE_ARGSECOND_PROBAS",
        "STATE_ARGTHIRD_PROBAS",
        "STATE_DISTANCES_LAB",
        "STATE_DISTANCES_UNLAB",
    ],
)
parser.add_argument("--DISTANCE_METRIC", default="euclidean")
parser.add_argument("--PRE_SAMPLING_METHOD", default="furthest")

parser.add_argument("--PRE_SAMPLING_ARG", type=int, default=10)


# FIXME wenn HYBRID -> HYBRID namen so ändern, dass die Werte von oben an den titel angefügt werden
config = parser.parse_args()


if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

config.HPC_WS_PATH = config_parser.get("HPC", "WS_PATH")
config.HPC_DATASETS_PATH = config_parser.get("HPC", "DATASET_PATH")
config.HPC_SSH_LOGIN = config_parser.get("HPC", "SSH_LOGIN")
config.HPC_OUTPUT_PATH = config_parser.get("HPC", "OUTPUT_PATH")

config.LOCAL_DATASETS_PATH = config_parser.get("LOCAL", "DATASET_PATH")
config.LOCAL_CODE_PATH = config_parser.get("LOCAL", "LOCAL_CODE_PATH")
config.LOCAL_OUTPUT_PATH = config_parser.get("LOCAL", "OUTPUT_PATH")


if config.WITH_CLASSICS or config.WITH_TUD_EVAL or config.WITH_PLOTS:
    print("config option deprecated")
    exit(-1)


config.EXPERIMENT_LAUNCH_SCRIPTS = (
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/" + config.EXP_TITLE
)

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
#SBATCH --output {{HPC_WS_PATH}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_WS_PATH}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_error.txt
{% if array %}#SBATCH --array {{START}}-{{END}}{% endif %}

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module load Python/3.8.6

{% if array %}i=$(( {{ OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ ITERATIONS_PER_BATCH }} )){% endif %}

MPLCONFIGPATH={{HPC_WS_PATH}}/cache python3 -m pipenv run python {{HPC_WS_PATH}}/code/{{PYTHON_FILE}}.py {{ CLI_ARGS }} {% if APPEND_OUTPUT_PATH %} {{ OUTPUT_PATH }}/{{ EXP_TITLE }} {% endif %}
exit 0
"""
)

bash_mode_common_template = Template("{{PYTHON_FILE}}.py {{ CLI_ARGS }}")

submit_jobs = Template(
    """#!/bin/bash
cd {{ HPC_WS_PATH }}/code
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
create_synthetic_training_data_id=$(sbatch --parsable {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/01_create_synthetic_training_data.slurm)
{%if WITH_HYPER_SEARCH %}hyper_search_id=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/02_hyper_search.slurm){% endif %}
train_imital_id=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id{%if WITH_HYPER_SEARCH %}:$hyper_search_id{% endif %} {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/03_train_imital.slurm)
alipy_eva=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id:$train_imital_id {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/05_alipy_eva.slurm)
exit 0
"""
)


sync_to_taurus = Template(
    """
    """
)


def write_slurm_and_bash_file(OUTPUT_FILE: str, APPEND_OUTPUT=False, **kwargs):
    with open(
        config.EXPERIMENT_LAUNCH_SCRIPTS + "/" + OUTPUT_FILE + ".slurm", "w"
    ) as f:
        content = slurm_common_template.render(
            OUTPUT_PATH=config.HPC_OUTPUT_PATH,
            APPEND_OUTPUT_PATH=APPEND_OUTPUT,
            EXP_TITLE=config.EXP_TITLE,
            **kwargs
        )
        f.write(content)
    with open(config.EXPERIMENT_LAUNCH_SCRIPTS + "/" + OUTPUT_FILE + ".tmp", "w") as f:
        content = bash_mode_common_template.render(**kwargs)
        if APPEND_OUTPUT:
            content += " " + config.LOCAL_OUTPUT_PATH + "/" + config.EXP_TITLE
        f.write(content)


if not os.path.exists(config.EXPERIMENT_LAUNCH_SCRIPTS):
    os.makedirs(config.EXPERIMENT_LAUNCH_SCRIPTS)


WS_CONFIG_OPTIONS = ""
if config.WS_MODE:
    WS_CONFIG_OPTIONS += " --WS_MODE"

    if config.USE_WS_LABELS_CONTINOUSLY:
        WS_CONFIG_OPTIONS += " --USE_WS_LABELS_CONTINOUSLY"


if not config.ONLY_ALIPY:
    START = 0
    END = int(config.TRAIN_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1

    write_slurm_and_bash_file(
        OUTPUT_FILE="01_create_synthetic_training_data",
        HPC_WS_PATH=config.HPC_WS_PATH,
        TITLE=config.EXP_TITLE,
        PYTHON_FILE="01_create_synthetic_training_data",
        array=True,
        START=START,
        END=END,
        ITERATIONS_PER_BATCH=config.ITERATIONS_PER_BATCH,
        OFFSET=0,
        CLI_ARGS=" "
        # p+ str(config.BATCH_MODE)
        + " --PRE_SAMPLING_METHOD "
        + str(config.PRE_SAMPLING_METHOD)
        + " --PRE_SAMPLING_ARG "
        + str(config.PRE_SAMPLING_ARG)
        + " --TOTAL_BUDGET "
        + str(config.TOTAL_BUDGET)
        + " --NR_LEARNING_SAMPLES "
        + str(config.ITERATIONS_PER_BATCH)
        + " --PRE_SAMPLING_HYBRID_UNCERT "
        + str(config.PRE_SAMPLING_HYBRID_UNCERT)
        + " --PRE_SAMPLING_HYBRID_PRED_UNITY "
        + str(config.PRE_SAMPLING_HYBRID_PRED_UNITY)
        + " --PRE_SAMPLING_HYBRID_FURTHEST "
        + str(config.PRE_SAMPLING_HYBRID_FURTHEST)
        + " --PRE_SAMPLING_HYBRID_FURTHEST_LAB "
        + str(config.PRE_SAMPLING_HYBRID_FURTHEST_LAB)
        + " "
        + " ".join(["--" + sa for sa in config.STATE_ARGS])
        + WS_CONFIG_OPTIONS
        + " --RANDOM_ID_OFFSET $i"
        + " --ANDREAS "
        + config.ANDREAS
        + " --DISTANCE_METRIC "
        + str(config.DISTANCE_METRIC)
        + " --OUTPUT_PATH ",
        APPEND_OUTPUT=True,
    )


if config.WITH_HYPER_SEARCH:
    write_slurm_and_bash_file(
        OUTPUT_FILE="02_hyper_search",
        HPC_WS_PATH=config.HPC_WS_PATH,
        TITLE=config.EXP_TITLE,
        PYTHON_FILE="02_hyper_search_or_train_imital",
        array=False,
        THREADS=24,
        MEMORY=5250,
        CLI_ARGS="--PERMUTATE_NN_TRAINING_INPUT "
        + str(config.PERMUTATE_NN_TRAINING_INPUT)
        + " --STATE_ENCODING listwise --TARGET_ENCODING "
        + config.TARGET_ENCODING
        + " --HYPER_SEARCH --N_ITER 100 "
        + " --OUTPUT_PATH ",
        APPEND_OUTPUT=True,
    )

if config.WITH_HYPER_SEARCH:
    hypered_appendix = " --HYPER_SEARCHED"
else:
    hypered_appendix = ""

if not config.ONLY_ALIPY:
    write_slurm_and_bash_file(
        OUTPUT_FILE="03_train_imital",
        HPC_WS_PATH=config.HPC_WS_PATH,
        TITLE=config.EXP_TITLE,
        PYTHON_FILE="03_train_imital",
        array=False,
        THREADS=20,
        MEMORY=5250,
        CLI_ARGS=hypered_appendix
        + " --PERMUTATE_NN_TRAINING_INPUT "
        + str(config.PERMUTATE_NN_TRAINING_INPUT)
        + " --TARGET_ENCODING "
        + config.TARGET_ENCODING
        + " --OUTPUT_PATH ",
        APPEND_OUTPUT=True,
    )


if config.WITH_ALIPY:
    alipy_init_seeds_template = Template(
        """#!/bin/bash
# run locally!
python 04_alipy_init_seeds.py --EXP_OUTPUT_PATH {{ EXP_OUTPUT_PATH }} --OUTPUT_PATH {{ EXPERIMENT_LAUNCH_SCRIPTS }} --DATASET_IDS {{ DATASET_IDS }} --STRATEGY_IDS {{ STRATEGY_IDS }} --AMOUNT_OF_RUNS {{ AMOUNT_OF_EVAL_RUNS }} --NON_SLURM --SLURM_FILE_TO_UPDATE {{ EXPERIMENT_LAUNCH_SCRIPTS }}/06_run_code_locally_as_bash.sh
python 04_alipy_init_seeds.py --EXP_OUTPUT_PATH {{ EXP_OUTPUT_PATH }} --OUTPUT_PATH {{ EXPERIMENT_LAUNCH_SCRIPTS }} --DATASET_IDS {{ DATASET_IDS }} --STRATEGY_IDS {{ STRATEGY_IDS }}  --AMOUNT_OF_RUNS {{ AMOUNT_OF_EVAL_RUNS }} --SLURM_FILE_TO_UPDATE {{ EXPERIMENT_LAUNCH_SCRIPTS }}/{{ SLURM_FILE_TO_UPDATE }}
    """
    )

    # if this file is run, it automatically updates the ARRAY indices for the alipy slurm job based on the result of this python script
    with open(config.EXPERIMENT_LAUNCH_SCRIPTS + "/04_alipy_init_seeds.sh", "w") as f:
        START = 0
        END = int(config.TEST_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1
        f.write(
            alipy_init_seeds_template.render(
                EXPERIMENT_LAUNCH_SCRIPTS=config.EXPERIMENT_LAUNCH_SCRIPTS,
                DATASET_IDS=",".join([str(id) for id in config.EVA_DATASET_IDS]),
                STRATEGY_IDS=",".join([str(id) for id in config.EVA_STRATEGY_IDS]),
                AMOUNT_OF_EVAL_RUNS=config.TEST_NR_LEARNING_SAMPLES,
                SLURM_FILE_TO_UPDATE="05_alipy_eva.slurm",
                EXP_OUTPUT_PATH=config.LOCAL_OUTPUT_PATH + "/" + config.EXP_TITLE,
            )
        )
    st = os.stat(config.EXPERIMENT_LAUNCH_SCRIPTS + "/04_alipy_init_seeds.sh")

    os.chmod(
        config.EXPERIMENT_LAUNCH_SCRIPTS + "/04_alipy_init_seeds.sh",
        st.st_mode | stat.S_IEXEC,
    )

    write_slurm_and_bash_file(
        OUTPUT_FILE="05_alipy_eva",
        HPC_WS_PATH=config.HPC_WS_PATH,
        TITLE=config.EXP_TITLE,
        PYTHON_FILE="05_alipy_eva",
        array=True,
        START="0",
        END="XXX",
        THREADS=2,
        MEMORY=2583,
        ITERATIONS_PER_BATCH=1,
        OFFSET=0,
        CLI_ARGS=" "
        + " --DATASETS_PATH "
        + config.HPC_DATASETS_PATH
        + " --OUTPUT_PATH "
        + config.HPC_OUTPUT_PATH
        + "/"
        + config.EXP_TITLE
        + " --RANDOM_SEEDS_INPUT_FILE "
        + config.EXPERIMENT_LAUNCH_SCRIPTS
        + "/04_random_seeds__slurm.csv --INDEX $SLURM_ARRAY_TASK_ID"
        + WS_CONFIG_OPTIONS,
    )


with open(config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_start_slurm_jobs.sh", "w") as f:
    f.write(
        submit_jobs.render(
            HPC_WS_PATH=config.HPC_WS_PATH,
            EXPERIMENT_LAUNCH_SCRIPTS=config.EXPERIMENT_LAUNCH_SCRIPTS,
            TITLE=config.EXP_TITLE,
            WITH_HYPER_SEARCH=config.WITH_HYPER_SEARCH,
            WITH_ALIPY=config.WITH_ALIPY,
        )
    )

st = os.stat(config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_start_slurm_jobs.sh")
os.chmod(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_start_slurm_jobs.sh",
    st.st_mode | stat.S_IEXEC,
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
    list(glob.glob(str(config.EXPERIMENT_LAUNCH_SCRIPTS) + "/*.tmp")),
    key=lambda v: sort_order[v.split("/")[-1]],
):
    with open(tmp_file, "r") as f:
        content = f.read()
        content = content.replace("$i", "0")
        if tmp_file.endswith("01_create_synthetic_training_data.tmp"):
            content = content.replace(
                "--NR_LEARNING_SAMPLES 10",
                "--NR_LEARNING_SAMPLES " + str(config.TRAIN_NR_LEARNING_SAMPLES),
            )

        if tmp_file.endswith("05_alipy_eva.tmp"):
            submit_content += (
                "python 05_ali_bash_parallel_runner_script.py --OUTPUT_PATH "
                + config.LOCAL_OUTPUT_PATH
                + "/"
                + config.EXP_TITLE
                + " --N_PARALLEL_JOBS "
                + str(config.N_JOBS)
                + " --DATASETS_PATH "
                + config.LOCAL_DATASETS_PATH
                + " --RANDOM_SEEDS_PATH "
                + config.EXPERIMENT_LAUNCH_SCRIPTS
                + " --N_TASKS XXX"
            )
        else:
            submit_content += "python " + content + "\n"
    os.remove(tmp_file)
with open(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_run_code_locally_as_bash.sh", "w"
) as f:
    f.write(submit_content)
st = os.stat(config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_run_code_locally_as_bash.sh")
os.chmod(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/06_run_code_locally_as_bash.sh",
    st.st_mode | stat.S_IEXEC,
)

with open(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_locally.sh", "w"
) as f:
    f.write(
        Template(
            """
# 6. 06_sync_and_run_experiment.sh
{{ EXPERIMENT_LAUNCH_SCRIPTS }}/04_alipy_init_seeds.sh
{{ EXPERIMENT_LAUNCH_SCRIPTS }}/06_run_code_locally_as_bash.sh
    """
        ).render(EXPERIMENT_LAUNCH_SCRIPTS=config.EXPERIMENT_LAUNCH_SCRIPTS)
    )
st = os.stat(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_locally.sh"
)
os.chmod(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_locally.sh",
    st.st_mode | stat.S_IEXEC,
)


with open(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_slurm.sh", "w"
) as f:
    f.write(
        Template(
            """
# 6. 06_sync_and_run_experiment.sh
# check if data can be downloaded from taurus
# updates taurus
# start experiment there
{{ EXPERIMENT_LAUNCH_SCRIPTS }}/04_alipy_init_seeds.sh
rsync -avz -P {{ LOCAL_CODE_PATH }} {{ HPC_SSH_LOGIN }}:{{ SLURM_PATH }}
ssh {{ HPC_SSH_LOGIN }} '{{ SLURM_PATH}}/code/{{ EXPERIMENT_LAUNCH_SCRIPTS }}/06_start_slurm_jobs.sh'
    """
        ).render(
            EXPERIMENT_LAUNCH_SCRIPTS=config.EXPERIMENT_LAUNCH_SCRIPTS,
            LOCAL_CODE_PATH=config.LOCAL_CODE_PATH,
            HPC_SSH_LOGIN=config.HPC_SSH_LOGIN,
            SLURM_PATH=config.HPC_WS_PATH,
        )
    )
st = os.stat(config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_slurm.sh")
os.chmod(
    config.EXPERIMENT_LAUNCH_SCRIPTS + "/07_sync_and_run_experiment_slurm.sh",
    st.st_mode | stat.S_IEXEC,
)
