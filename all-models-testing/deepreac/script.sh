s=$1
p=$2
i=$3
j=$4
g=$5

train="../dataset/sc/train/NSi"
train+="_FullCV_$j.csv"
val="../dataset/sc/val/NSi"
val+="_FullCV_$j.csv"
test="../dataset/sc/test/NSi"
test+="_FullCV_$j.csv"
ckpt="../perm_results/deepreac/sc"
ckpt+="_$j"
ckpt+="_test_chemprop_h300_bs8.csv"
pth="/../models/deepreac/sci"
pth+="_$j"
pth+="_test_chemprop_h300_bs8.pt"
pred="../predict/deepreac/sci"
pred+="_$j"
pred+="_test_chemprop_h300_bs8.csv"

python DeepReac_train.py --train_file $train --val_file $val --test_file $test --hidden_size 300 --epochs 100 --batch_size 8 --stats_file $ckpt --model_path $pth --pred_file $pred --device 1 --seed $s --pytorch_seed $p
