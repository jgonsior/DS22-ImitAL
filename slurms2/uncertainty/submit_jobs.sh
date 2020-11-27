#!/bin/bash
create_ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/uncertainty/create_ann_training_data.slurm)
create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$create_ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/uncertainty/create_ann_eval_data.slurm)
classics_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/uncertainty/classics.slurm)
plots_id=$(sbatch --parsable --dependency=afterok:$create_ann_eval_data:$create_ann_eval_idk:$classics_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/uncertainty/plots.slurm)
exit 0
