#!/bin/bash
 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/trained_mlp --NN_BINARY tmp/_vd_true_100_rf_true_-1_h_false_nsp_true_chs_true_2_nqpi_true_samar_true_MLP_gn_false/trained_ann.pickle --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE --REPRESENTATIVE_FEATURES --CLASSIFIER MLP
# done
 
for i in {1..10}
do
    python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/unc_mlp --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE --CLASSIFIER MLP
done
# 
# 
# 
# for i in {1..10}
# do
#     python imit_training.py --DATASETS_PATH ../datasets --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/opti_mlp --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --NR_LEARNING_ITERATIONS 1 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE --CLASSIFIER MLP
# done
# 
# 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/trained --NN_BINARY tmp/_vd_tru..10_rf_true_-1_h_false_nsp_true_chs_true_2_5/trained_ann.pickle --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE --REPRESENTATIVE_FEATURES
# done
#  
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/trained_new  --NN_BINARY tmp/_vd_tru..10_rf_true_-1_h_false_nsp_true_chs_true_2_5/trained_ann.pickle --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --HYPERCUBE --REPRESENTATIVE_FEATURES
# done
# 
# 
# 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/random --SAMPLING random --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE
# done
# 
# 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/unc --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --NEW_SYNTHETIC_PARAMS --HYPERCUBE
# done
# 
# 
# for i in {1..10}
# do
#     python imit_training.py --DATASETS_PATH ../datasets --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/opti_new --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --AMOUNT_OF_PEAKED_OBJECTS 20 --MAX_AMOUNT_OF_WS_PEAKS 0 --NR_LEARNING_ITERATIONS 20 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --HYPERCUBE
# done
# 
# 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/random_new --SAMPLING random --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --HYPERCUBE
# done
# 
# 
# for i in {1..10}
# do
#     python single_al_cycle.py --OUTPUT_DIRECTORY ../datasets/il_training_data/plot_evolutions/unc_new --SAMPLING uncertainty_max_margin --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 20 --RANDOM_SEED $i --N_JOBS 8 --PLOT_EVOLUTION --AMOUNT_OF_FEATURES 2 --NR_LEARNING_ITERATIONS 20 --CONVEX_HULL_SAMPLING --VARIABLE_INPUT_SIZE --HYPERCUB
# done
# 
# 
