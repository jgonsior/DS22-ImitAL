#!/bin/bash
for i in $(seq 1 50);
do 
    python single_al_cycle.py --OUTPUT_DIRECTORY mm2  --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
    # python single_al_cycle.py --OUTPUT_DIRECTORY random2  --SAMPLING random --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
    # python single_al_cycle.py --OUTPUT_DIRECTORY lc  --SAMPLING uncertainty_lc --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
    # python single_al_cycle.py --OUTPUT_DIRECTORY entropy  --SAMPLING uncertainty_entropy --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
done
