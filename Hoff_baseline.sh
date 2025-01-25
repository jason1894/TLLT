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

Times=($(seq 1 1 3))


echo "office A2D with pretrained weight"
python office.py --source_domain "A" --target_domain "D" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 1.0 --seed 2324 --pretrained

echo "office A2W with pretrained weight"
python office.py --source_domain "A" --target_domain "W" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.4 --seed 2324 --pretrained

echo "office D2A with pretrained weight"
python office.py --source_domain "D" --target_domain "A" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.2 --seed 2324 --pretrained

echo "office D2W with pretrained weight"
python office.py --source_domain "D" --target_domain "W" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.6 --seed 2324 --pretrained

echo "office W2A with pretrained weight"
python office.py --source_domain "W" --target_domain "A" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.4 --lr 0.5 --seed 2324 --pretrained

echo "office W2D with pretrained weight"
python office.py --source_domain "W" --target_domain "D" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.5 --seed 2324 --pretrained


echo "office ACW2D with pretrained weight"
python office.py --source_domain "A" "C" "W" --target_domain "D" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.1 --seed 2324 --pretrained

echo "office ACD2W"
python office.py --source_domain "A" "C" "D" --target_domain "W" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.4 --seed 2324 --pretrained

echo "office CDW2A with pretrained weight"
python office.py --source_domain "C" "D" "W" --target_domain "A" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.4 --lr 0.1 --seed 2324 --pretrained

echo "office ADW2C with pretrained weight"
python office.py --source_domain "A" "D" "W" --target_domain "C" --data_name "office" --bths 64 --epochs 200 --time 1 --Method "Method_A" --latdim 128 --r 0.1 --lr 0.2 --seed 2324 --pretrained

