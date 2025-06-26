# YieldNet
Yield prediction with continuous surrogate TS (CGR) computation

## Training from scratch
To train the models, run the following
```sh
pip install -r requirements.txt
cd YieldNet
```
for single step reaction use `run_one.sh` and for multistep use `run_multi.sh`. Both these files need 5 set of arguments: master_seed (for training data shuffling), pytorch_seed (to get the same initialization), data_id (for example, for NS0 it's 25, for GP it's 0, etc.), split_id and device_id. For better clarity, it's recommended to watch those files. All the arguments provided in those files are used to get our main trained model. Modify accordingly to train your datasets. You can use `rand_run.py` with our default running seed values to get the trained models to reproduce the result.
```sh
python rand_run.py device_id
```
For example, 
```sh
python rand_run.py 0
```
Further to know more, follow `new-README.md` inside that directory.

## Instructions
To run the models (loading the saved checkpoint and running test evaluation), first download
the models folders from the links below and place the folder inside all-models-testing. After
extracting both the folders (and placing in a common folder), the final structure should be
like:

```
all-models-testing  
|-- ...  
|-- models  
    |-- chemprop  
        |-- dfs10  
        |-- dfs75  
        |-- ...  
    |-- deepreac  
        |-- dfs10  
        |-- dfs75  
        |-- ...  
    |-- ...  
    |-- mpnn  
        |-- dfs10  
        |-- dfs75  
        |-- ... 
```

Following this, the load-predict.py script can be run (after setting up the python environment
as per requirements.txt using pip or conda) using:

```sh
python load-predict.py --gpu device_id
```

## TS visualization
Code for generating the raw networkx graph is given below
```sh
cd ts_vis
```
Then run the `ts_vis.ipynb` to get the networkx graphs.

## Drive link for model files
Zip file for the trained model folder is given below
https://rebrand.ly/tynet

## Names
The datasets are named as follows: NS0 = nss25, NS1 = nss40, NS2 = nss50, NS3 = nss60,
NS4 = nss75, DF0 = dfs10, DF1 = dfs75, DF2 = dfs90, GP = gpy0, GP1 = gpy1, GP2 = gpy2,
SC = sc, DF = df, NS = ns, USPTO = uspto.

The models are named as follows: GCN = mpnn, HGT = g1, TAG = g2, GIN = g4,
YieldNet = chemprop.

## Datasets
Datasets are inside `./YieldNet/dataset/` folder.
