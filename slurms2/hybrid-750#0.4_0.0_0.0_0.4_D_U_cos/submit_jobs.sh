#!/bin/bash
ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid-750#0.4_0.0_0.0_0.4_D_U_cos/ann_training_data.slurm)
train_ann_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/hybrid-750#0.4_0.0_0.0_0.4_D_U_cos/train_ann.slurm)




exit 0