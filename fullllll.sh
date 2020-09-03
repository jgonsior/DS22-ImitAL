#!/bin/bash

# python full_experiment.py --RANDOM_SEED 1 --LOG_FILE log.txt --OUTPUT_DIRECTORY tmp/ --TRAIN_CLASSIFIER MLP --TRAIN_GENERATE_NOISE --TEST_GENERATE_NOISE --TRAIN_NR_LEARNING_SAMPLES 2000 --TEST_NR_LEARNING_SAMPLES 2000 --FINAL_PICTURE tmp/plots_2000_TRAIN_CLASSIFIER_TRAIN_GENERATE_NOISE_TEST_GENERATE_NOISE/_MLP_True_True
python fuller_experiment.py --TRAIN_STATE_DIFF_PROBAS True
python fuller_experiment.py --TRAIN_STATE_ARGSECOND_PROBAS True
python fuller_experiment.py --TRAIN_STATE_ARGTHIRD_PROBAS True
python fuller_experiment.py --TRAIN_STATE_DISTANCES True
python fuller_experiment.py --TRAIN_STATE_NO_LRU_WEIGHTS True
python fuller_experiment.py --TRAIN_STATE_LRU_AREAS_LIMIT 0,1,3,5,10,20
python fuller_experiment.py --TRAIN_STATE_NO_LRU_WEIGHTS True --TRAIN_STATE_LRU_AREAS_LIMIT 0,1,5,10,20 --TRAIN_STATE_DIFF_PROBAS True
python fuller_experiment.py --TRAIN_CLASSIFIER MLP,RF,SVM --TEST_CLASSIFIER MLP,RF,SVM
python fuller_experiment.py --TRAIN_GENERATE_NOISE True --TEST_GENERATE_NOISE True
python fuller_experiment.py --NR_QUERIES_PER_ITERATION 1,5,10
python fuller_experiment.py --TRAIN_VARIABLE_DATASET True
python fuller_experiment.py --TRAIN_VARIANCE_BOUND 1,2,5
python fuller_experiment.py --TRAIN_HYPERCUBE True
python fuller_experiment.py --TRAIN_NEW_SYNTHETIC_PARAMS True --TEST_NEW_SYNTHETIC_PARAMS True
python fuller_experiment.py --TRAIN_CONVEX_HULL_SAMPLING True
python fuller_experiment.py --TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED True
