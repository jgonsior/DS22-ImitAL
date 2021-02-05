#!/bin/bash
ann_training_data_id=$(sbatch --parsable /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/single_full_cos_nrf/ann_training_data.slurm)
hyper_search_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/single_full_cos_nrf/hyper_search.slurm)
train_ann_id=$(sbatch --parsable --dependency=afterok:$ann_training_data_id:$hyper_search_id /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurms2/single_full_cos_nrf/train_ann.slurm)



exit 0