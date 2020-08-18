#!/bin/bash
for j in new_old_fixed #100000
do
    echo $j
    for i in $(seq 1 50);
    do
        python single_al_cycle.py --NN_BINARY $j.pickle --OUTPUT_DIRECTORY $j.csv --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
    done

    # sed -i 's/trained_nn/$j/g' $j/creation.csv
done
