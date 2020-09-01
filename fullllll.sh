#!/bin/bash

# python full_experiment.py --RANDOM_SEED 1 --LOG_FILE log.txt --OUTPUT_DIRECTORY tmp/ --TRAIN_CLASSIFIER MLP --TRAIN_GENERATE_NOISE --TEST_GENERATE_NOISE --TRAIN_NR_LEARNING_SAMPLES 2000 --TEST_NR_LEARNING_SAMPLES 2000 --FINAL_PICTURE tmp/plots_2000_TRAIN_CLASSIFIER_TRAIN_GENERATE_NOISE_TEST_GENERATE_NOISE/_MLP_True_True
python fuller_experiment.py --TRAIN_LRU_AREAS_LIMIT 1,3,5 --TEST_LRU_AREAS_LIMIT 1,3,5
python fuller_experiment.py --TRAIN_NO_DIFF_FEATURES True --TEST_NO_DIFF_FEATURES True
python fuller_experiment.py --TRAIN_CLASSIFIER MLP,RF,SVM --TEST_CLASSIFIER MLP,RF,SVM
python fuller_experiment.py --TRAIN_CLASSIFIER MLP,RF,SVM --TEST_CLASSIFIER MLP,RF,SVM
python fuller_experiment.py --TRAIN_GENERATE_NOISE True --TEST_GENERATE_NOISE True
python fuller_experiment.py --TRAIN_REPRESENTATIVE_FEATURES True --TEST_REPRESENTATIVE_FEATURES True
python fuller_experiment.py --NR_QUERIES_PER_ITERATION 1,5
python fuller_experiment.py --TRAIN_VARIABLE_DATASET True
python fuller_experiment.py --TRAIN_VARIANCE_BOUND 1,2,5
python fuller_experiment.py --TRAIN_HYPERCUBE True
python fuller_experiment.py --TRAIN_NEW_SYNTHETIC_PARAMS True --TEST_NEW_SYNTHETIC_PARAMS True
python fuller_experiment.py --TRAIN_CONVEX_HULL_SAMPLING True
python fuller_experiment.py --TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED True
