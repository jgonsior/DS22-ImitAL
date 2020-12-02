#!/bin/bash
SLURM_ARRAY_TASK_ID=$1
i=$(( 100000 + $SLURM_ARRAY_TASK_ID * 10 ))
echo $i
