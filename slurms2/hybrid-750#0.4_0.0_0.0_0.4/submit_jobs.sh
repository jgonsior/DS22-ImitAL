#!/bin/bash
ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid-750#0.4_0.0_0.0_0.4/ann_training_data.slurm)
train_ann_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid-750#0.4_0.0_0.0_0.4/train_ann.slurm)
hyper_search_id=$(sbatch --parsable --dependency=afterok:$train_ann_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid-750#0.4_0.0_0.0_0.4/hyper_search.slurm)
create_ann_eval_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$train_ann_id:$hyper_search_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/hybrid-750#0.4_0.0_0.0_0.4/ann_eval_data.slurm)

plots_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$create_ann_eval_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal//slurms2/hybrid-750#0.4_0.0_0.0_0.4/plots.slurm)
exit 0