#!/bin/bash

#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH -J preoffice
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err

#echo Time is `date`
#echo Directory is $PWD
#echo This job runs on the following nodes: $SLURM_JOB_NODELIST
#echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

Methods=("Method_A" "Method_C" "Method_B" "Method_D")
Times=($(seq 1 1 10))


echo "BreastCancer_normalize"
for method in "${Methods[@]}";do
    for time in "${Times[@]}";do
        python toy_class.py --data_name "BreastCancer_normalize" --epochs 200  --depth 20 --time ${time} --Method ${method} --latdim 6 --r 0.1 --lr 3.0
    done
done 

