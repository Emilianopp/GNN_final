#!/bin/bash
#SBATCH --time=5:0:0
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --account=
#SBATCH --output ./cit1epochs.000001%J.out # STDOUT


cd /home/emiliano/projects/def-cbravo/emiliano/DyGLib/
source ~/TGN/bin/activate


python  train_snapshot.py --dataset_name Citation --model_name TGAT --load_best_configs --num_runs 1 --gpu 1 --num_epochs 2 --learning_rate .0001

python  produce_embeddings.py --dataset_name Citation --model_name TGAT --load_best_configs --num_runs 1 --gpu 1 --num_epochs 2 --learning_rate .0001



