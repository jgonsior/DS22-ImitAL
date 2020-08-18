#!/bin/bash
for i in $(seq 1 50);
do 
    python single_al_cycle.py --NN_BINARY ../datasets/il_training_data/fixed_synthetic_5_50_20_large/trained_listwise_binary.pickle --OUTPUT_DIRECTORY ../datasets/il_training_data/polytope_hard_nn.csv --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1&
done
