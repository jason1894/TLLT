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

# moon: latdim 200, r 0.9. latdim 100, r 1.0. latdim 50, r 0.5
# circles: latdim 200, r 0.9. latdim 100, r 0.2,  latdim 50, r 0.1
echo "toys"
for method in "${Methods[@]}";do
    for time in "${Times[@]}";do
        python toy_class.py --data_name "circles" --bths 64 --epochs 200  --depth 20 --time ${time} --Method ${method} --latdim 2 --r 0.9 --lr 0.5
    done
done 

for method in "${Methods[@]}";do
    for time in "${Times[@]}";do
        python toy_class.py --data_name "moon" --bths 64 --epochs 200  --depth 20 --time ${time} --Method ${method} --latdim 2 --r 0.4 --lr 3.0
    done
done 