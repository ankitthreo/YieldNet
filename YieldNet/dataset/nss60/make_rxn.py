import pandas as pd

for i in range(1, 11):
    sheet_id = str(i).zfill(2)
    df_ = pd.read_excel("NSS60_data.xlsx", sheet_name='FullCV_'+sheet_id)
    rxns = []
    for j, row in df_.iterrows():
        cpa = row['CPA']
        sub = row['Substrate']
        thiol = row['Thiol']
        int1 = row['active_substrate']
        int2 = row['pre_product']
        pdt = row['product']
        
        step1 = f"{sub}.{cpa}>>{int1}"
        step2 = f"{int1}.{thiol}>>{int2}"
        step3 = f"{int2}>>{pdt}"
        rxn = f"{step1}|{step2}|{step3}"
        rxns.append(rxn)
        
    df_['rxn'] = rxns
    df__ = df_[['rxn', 'ee']]
    df = df__.drop_duplicates()
    train, val, test = df.iloc[:210], df.iloc[210:240], df.iloc[240:]
    df.to_csv(f"./full/NS_FullCV_{str(i)}.csv", index=False)
    train.to_csv(f"./train/NS_FullCV_{str(i)}.csv", index=False)
    val.to_csv(f"./val/NS_FullCV_{str(i)}.csv", index=False)
    test.to_csv(f"./test/NS_FullCV_{str(i)}.csv", index=False)
