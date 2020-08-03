#!/bin/bash
for i in $(seq 1 2);
do 
    for j in $(seq 1 7);
    do
        # echo $(bash -c 'RANDOM='$((42+$i+$j*100))'; echo "$RANDOM"')&
        python single_al_cycle.py --OUTPUT_DIRECTORY $1  --SAMPLING $2 --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $(bash -c 'RANDOM='$((42+$i+$j*100))'; echo "$RANDOM"') --N_JOBS 1&
    done
        # echo $(bash -c 'RANDOM='$((42+$i+$j*100+100))'; echo "$RANDOM"')
        python single_al_cycle.py --OUTPUT_DIRECTORY $1 --SAMPLING $2 --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $(bash -c 'RANDOM='$((42+$i+$j*100+100))'; echo "$RANDOM"') --N_JOBS 1
done
