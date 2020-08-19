#!/bin/bash
python full_experiment.py \
    --OUTPUT_DIRECTORY "tmp/" \
    --TRAIN_VARIABLE_DATASET \
    --TRAIN_NR_LEARNING_SAMPLES 10 \
    --TRAIN_AMOUNT_OF_FEATURES -1 \
    --TRAIN_HYPERCUBE \
    --TRAIN_OLD_SYNTHETIC_PARAMS \
    --TEST_VARIABLE_DATASET \
    --TEST_NR_LEARNING_SAMPLES 10 \
    --TEST_AMOUNT_OF_FEATURES -1 \
    --TEST_HYPERCUBE \
    --TEST_OLD_SYNTHETIC_PARAMS \
    --TEST_COMPARISONS random uncertainty_max_margin 
