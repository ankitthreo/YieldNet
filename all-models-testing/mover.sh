#!/bin/bash

data="$1"
hidden="$2"
path="./yieldbert-20/results/$data/" 
save="./models/yieldbert/$data/" 

mkdir -p $save

for i in {1..10}
do
    loc="cv$i"
    loc+="/best_model"
    cp -r $path$loc $save/cv$i-h$hidden-model
done
