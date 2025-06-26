#!/bin/bash

s=$1
p=$2
i=$3
j=$4
g=$5

train="./dataset/gpy$i/train/GP$i"
train+="_FullCV_$j.csv"
val="./dataset/gpy$i/val/GP$i"
val+="_FullCV_$j.csv"
test="./dataset/gpy$i/test/GP$i"
test+="_FullCV_$j.csv"
ckpt="./perm_results/gold_perm/gpy$i/reg0.1/cv$j"
ckpt+="_ckpts_ds2s_h20_gold"

gold="./perm_results/soft_perm/gpy$i/reg0.1/cv$j"
gold+="_ckpts_ds2s_h20/fold_0/perms"

rm -rf $ckpt/*
cp -r val_perms/$j/* chemprop/train/val_perm

python train.py \
    --data_path $train \
    --separate_val_path $val \
    --separate_test_path $test \
    --save_dir $ckpt \
    --gpu $g \
    --dataset_type regression \
    --metric mae \
    --extra_metrics rmse r2 \
    --reaction --reaction_mode reac_diff \
    --explicit_h \
    --perm_type hard \
    --perm_mode gold \
    --gold_path $gold \
    --gold_pname gold \
    --loss_agg_func mean \
    --epochs 100 \
    --init_lr 0.0001\
    --max_lr 0.001 \
    --final_lr 0.0001 \
    --aggregation mean \
    --step_aggregation sum \
    --seed $s \
    --batch_size 50 \
    --ffn_hidden_size 20 \
    --hidden_size 20 \
    --pytorch_seed $p \
    # --use_indicator\
    # --use_node_tf\
    # --use_step_tf\
    # --step_tf NNAttention\
    # --aggregation set2set\

rm -rf chemprop/train/val_perm/*
