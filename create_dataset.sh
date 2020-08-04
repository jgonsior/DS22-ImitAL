#!/bin/bash
while :
do 
    for j in $(seq 1 8);
    do
        if [ "$j" -eq "8" ]; then
            python imit_training.py --DATASETS_PATH ../datasets  --OUTPUT_DIRECTORY tmp/ --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --START_SET_SIZE 1 --DATASET_NAME synthetic --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED -2 --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --LOG_FILE tmp/test.txt
        else
            python imit_training.py --DATASETS_PATH ../datasets  --OUTPUT_DIRECTORY tmp/ --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --START_SET_SIZE 1 --DATASET_NAME synthetic --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED -2 --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --LOG_FILE tmp/test.txt& 
        fi
    done
done
