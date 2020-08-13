#!/bin/bash
for i in $(seq 1 50);
do
    python imit_training.py --DATASETS_PATH ../datasets --OUTPUT_DIRECTORY ../repr_variable_synthetic_5_50_20_test --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --AMOUNT_OF_LEARN_ITERATIONS 1& 
done
