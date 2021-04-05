#!/bin/bash
python ann_training_data.py   --INITIAL_BATCH_SAMPLING_METHOD furthest --BASE_PARAM_STRING batch_single_full_unlab --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY test --TOTAL_BUDGET 10 --NR_LEARNING_SAMPLES 3 --INITIAL_BATCH_SAMPLING_HYBRID_UNCERT 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB 0.2 --STATE_ARGSECOND_PROBAS --STATE_DISTANCES_UNLAB  --RANDOM_ID_OFFSET 0 --DISTANCE_METRIC euclidean
python train_ann.py --OUTPUT_DIRECTORY test/ --BASE_PARAM_STRING batch_single_full_unlab
python classics.py --OUTPUT_DIRECTORY test/ --TOTAL_BUDGET 10 --NR_LEARNING_SAMPLES 3 --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --TEST_RANDOM_ID_OFFSET 0
python plots.py --OUTPUT_DIRECTORY test --TOTAL_BUDGET 10 --NR_LEARNING_SAMPLES 4 --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --BASE_PARAM_STRING batch_single_full_unlab --FINAL_PICTURE test/plots_batch_single_full_unlab/ --PLOT_METRIC acc_auc
