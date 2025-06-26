#!/bin/bash

epoch=100
s=$1
p=$2
i=$3
j=$4

train="../dataset/gpy$i/train/GP$i"
train+="_FullCV_$j.csv"
val="../dataset/gpy$i/val/GP$i"
val+="_FullCV_$j.csv"
test="../dataset/gpy$i/test/GP$i"
test+="_FullCV_$j.csv"
ckpt="./results/gpy$i/cv$j"

rm -rf $ckpt/*

python3 run_training.py \
    --experiment full \
    --base-model pretrained \
    --num-epochs $epoch \
    --lr 0.0001 \
    --gradient-accumulation-steps 1 \
    --seed $p \
    --batch-size 50 \
    --dropout 0.0 \
    --train-path $train \
    --val-path $val \
    --test-path $test \
    --ckpt $ckpt \

rm -rf $(find $ckpt -mindepth 1 -type d ! -name *-$epoch)

cp -r outputs/* $ckpt
# mv wandb/run*/output.log $ckpt
rm -rf cache_dir
rm -rf runs
rm -rf wandb
rm -rf outputs
