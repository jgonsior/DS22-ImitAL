#!/bin/bash
ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid/ann_training_data.slurm)
train_ann_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid/train_ann.slurm)
create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$train_ann_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/hybrid/ann_eval_data.slurm)
classics_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/hybrid/classics.slurm)
plots_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$create_ann_eval_id:$classics_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/hybrid/plots.slurm)
exit 0