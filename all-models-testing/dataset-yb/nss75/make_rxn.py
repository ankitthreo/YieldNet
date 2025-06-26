import pandas as pd

for i in range(1, 11):
    sheet_id = str(i).zfill(2)
    df_ = pd.read_excel("NSS75_data.xlsx", sheet_name='FullCV_'+sheet_id)
    rxns = []
    for j, row in df_.iterrows():
        cpa = row['CPA']
        sub = row['Substrate']
        thiol = row['Thiol']
        int1 = row['active_substrate']
        int2 = row['pre_product']
        pdt = row['product']
        
        rxn = f"{cpa}*{sub}*{thiol}*{int1}*{int2}*{pdt}"
        rxns.append(rxn)
        
    df_['reaction'] = rxns
    df_['temp'] = ''
    df_['output'] = df_['ee']
    df__ = df_[['reaction', 'output']]
    df = df__.drop_duplicates()
    train, val, test = df.iloc[:210], df.iloc[210:240], df.iloc[240:]
    df.to_csv(f"./full/NSi_FullCV_{str(i)}.csv", index=False)
    train.to_csv(f"./train/NSi_FullCV_{str(i)}.csv", index=False)
    val.to_csv(f"./val/NSi_FullCV_{str(i)}.csv", index=False)
    test.to_csv(f"./test/NSi_FullCV_{str(i)}.csv", index=False)
    