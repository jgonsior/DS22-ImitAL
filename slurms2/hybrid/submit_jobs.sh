#!/bin/bash
ann_training_data_id=$(sbatch --parsable ../datasets/short_test/imitating-weakal/slurms2/hybrid/ann_training_data.slurm)
train_ann_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id ../datasets/short_test/imitating-weakal/slurms2/hybrid/train_ann.slurm)

create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$train_ann_id ../datasets/short_test/imitating-weakal//slurms2/hybrid/ann_eval_data.slurm)
classics_id=$(sbatch --parsable ../datasets/short_test/imitating-weakal//slurms2/hybrid/classics.slurm)
plots_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$create_ann_eval_id:$classics_id ../datasets/short_test/imitating-weakal//slurms2/hybrid/plots.slurm)
exit 0