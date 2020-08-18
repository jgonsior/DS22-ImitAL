#!/bin/bash
for i in $(seq 1 100);
do 
    # python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/hypercube_uncert_mm.csv  --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
    python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/polytope_uncert_mm_flipped_class_sep.csv  --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
done
