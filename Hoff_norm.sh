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


# office: latdim 512, r 0.6. latdim 256, r 0.7. latdim 128, r 0.7.  latdim 64, r 0.1.
echo "office"
for method in "${Methods[@]}";do
    for time in "${Times[@]}";do
        python office.py --source_domain "C" "D" "W" --target_domain "A" --data_name "office" --bths 64 --epochs 200 --time ${time} --Method ${method} --latdim 128 --r 0.8 --lr 0.2
    done
done            
