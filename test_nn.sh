#!/bin/bash
RANDOM=42
python single_al_cycle.py --DATASETS_PATH ../datasets --OUTPUT_CSV $1 --DATA_PATH $2 --SAMPLING $3 --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $RANDOM --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0& 

1. fix das skript oben für den datensatz mit 40 future peaks, und für uncertainty unten
2. das hier auf taurus als array starten -> hyper_params.csv dateien für alle 5 uncertainty strategien erstellen, und für alle neuronalen netze die ich aktuell habe

# for i in $(seq 1 4);
# do 
#     for j in $(seq 1 7);
#     do
#         python single_al_cycle.py --DATASETS_PATH ../datasets --OUTPUT_CSV $1 --DATA_PATH $2 --SAMPLING $3 --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $RANDOM --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0& 
#     done
#         python single_al_cycle.py --DATASETS_PATH ../datasets --OUTPUT_CSV $1 --DATA_PATH $2 --SAMPLING $3 --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $RANDOM --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 
# done
