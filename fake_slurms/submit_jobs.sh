#!/bin/bash
python ann_training_data.py   --INITIAL_BATCH_SAMPLING_METHOD furthest --BASE_PARAM_STRING batch_single_full_10 --INITIAL_BATCH_SAMPLING_ARG 10 --OUTPUT_DIRECTORY test --USER_QUERY_BUDGET_LIMIT 50 --TRAIN_NR_LEARNING_SAMPLES 10 --INITIAL_BATCH_SAMPLING_HYBRID_UNCERT 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB 0.2 --STATE_ARGSECOND_PROBAS --STATE_ARGTHIRD_PROBAS --STATE_DISTANCES_LAB --STATE_DISTANCES_UNLAB  --TRAIN_RANDOM_ID_OFFSET 0 --DISTANCE_METRIC euclidean
python train_ann.py --OUTPUT_DIRECTORY test/ --BASE_PARAM_STRING batch_single_full_10
python ann_eval_data.py   --INITIAL_BATCH_SAMPLING_METHOD furthest --BASE_PARAM_STRING batch_single_full_10 --INITIAL_BATCH_SAMPLING_ARG 10 --OUTPUT_DIRECTORY test/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES 10 --INITIAL_BATCH_SAMPLING_HYBRID_UNCERT 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB 0.2 --STATE_ARGSECOND_PROBAS --STATE_ARGTHIRD_PROBAS --STATE_DISTANCES_LAB --STATE_DISTANCES_UNLAB  --TEST_RANDOM_ID_OFFSET 0 --DISTANCE_METRIC euclidean
python classics.py --OUTPUT_DIRECTORY test/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES 10 --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --TEST_RANDOM_ID_OFFSET 0
python plots.py --OUTPUT_DIRECTORY test --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES 10 --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --BASE_PARAM_STRING batch_single_full_10 --FINAL_PICTURE test/plots_batch_single_full_10/ --PLOT_METRIC acc_auc
