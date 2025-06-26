import pandas as pd
import numpy as np

root = 'gpt'


def splits(root, filename):
    
    df = pd.read_csv("./"+root+"/full/"+filename)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    try:
        df.drop('cluster_labels', inplace=True, axis=1)
    except KeyError:
        None
    lth =  len(df)
    train = df.iloc[:round(lth*0.7)]
    val = df.iloc[round(lth*0.7):round(lth*0.8)]
    test = df.iloc[round(lth*0.8):]
    train.to_csv("./"+root+"/train/"+filename, index=False)
    val.to_csv("./"+root+"/val/"+filename, index=False)
    test.to_csv("./"+root+"/test/"+filename, index=False)
    
    
for i in range(1, 11):
    filename = f"GPT_FullCV_{i}.csv"
    splits(root, filename)