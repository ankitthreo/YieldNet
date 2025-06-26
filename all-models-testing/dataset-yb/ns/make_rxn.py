import pandas as pd

for i in range(1, 11):
    sheet_id = str(i).zfill(2)
    df_ = pd.read_excel("denmark_intermediate_dataset.xlsx", sheet_name='FullCV_'+sheet_id)
    rxns = []
    for j, row in df_.iterrows():
        cpa = row['CPA']
        sub = row['Substrate']
        thiol = row['Thiol']
        int1 = row['active_substrate']
        int2 = row['pre_product']
        pdt = row['product']
        
        # step1 = f"{sub}.{cpa}>>{int1}"
        # step2 = f"{int1}.{thiol}>>{int2}"
        # step3 = f"{int2}>>{pdt}"
        # rxn = f"{step1}|{step2}|{step3}"
        rxn = f"{cpa}*{sub}*{thiol}*{int1}*{int2}*{pdt}"
        rxns.append(rxn)
        
    df_['reaction'] = rxns
    df_['output'] = df_['ee']
    df = df_[['reaction', 'output']]
    train, val, test = df.iloc[:719], df.iloc[719:822], df.iloc[822:]
    df.to_csv(f"./full/NSi_FullCV_{str(i)}.csv", index=False)
    train.to_csv(f"./train/NSi_FullCV_{str(i)}.csv", index=False)
    val.to_csv(f"./val/NSi_FullCV_{str(i)}.csv", index=False)
    test.to_csv(f"./test/NSi_FullCV_{str(i)}.csv", index=False)
