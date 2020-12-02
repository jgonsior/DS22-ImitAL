#!/bin/bash
create_ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/random/create_ann_training_data.slurm)
create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$create_ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/random/create_ann_eval_data.slurm)
classics_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/random/classics.slurm)
plots_id=$(sbatch --parsable --dependency=afterok:$create_ann_training_data_id:$create_ann_eval_id:$classics_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/random/plots.slurm)
exit 0
