import pandas as pd

for i in range(1, 11):
    sheet_id = str(i)#.zfill(2)
    df_ = pd.read_excel("Pd_nhc_base.xlsx", sheet_name='FullCV_'+sheet_id)
    rxns = []
    for j, row in df_.iterrows():
        Sub1 = row['Sub1']
        Sub2 = row['Sub2']
        Base = row['Base']
        Solvent = row['Solvent']
        Cat = row['active_catalyst_smiles_dative']
        int1 = row['new_smiles_dative']
        int2 = row['TM_smiles_dative']
        metal_halide = row['metal_halide']
        metal_boron_base = row['metal_boron_base']
        pdt = row['Product']
        
        #step1 = f"{Sub2}.{Cat}.{Solvent}>>{int1}.{Solvent}"
        #step2 = f"{Sub1}.{int1}.{Base}.{Solvent}>>{int2}.{metal_halide}.{metal_boron_base}.{Solvent}" #
        #step3 = f"{int2}.{Solvent}>>{pdt}.{Solvent}"
        
        #rxn = f"{step1}|{step2}|{step3}"
        rxn = f"{Sub1}*{Sub2}*{Cat}*{Base}*{Solvent}*{int1}*{int2}*{metal_halide}*{metal_boron_base}*{pdt}"
        rxns.append(rxn)
        
    df_['reaction'] = rxns
    df_['temp'] = ''
    df_['output'] = df_['Yield']
    df__ = df_[['reaction', 'output']]
    df = df__.drop_duplicates()
    train, val, test = df.iloc[:337], df.iloc[337:337+48], df.iloc[337+48:]
    df.to_csv(f"./full/SCi_FullCV_{str(i)}.csv", index=False)
    train.to_csv(f"./train/SCi_FullCV_{str(i)}.csv", index=False)
    val.to_csv(f"./val/SCi_FullCV_{str(i)}.csv", index=False)
    test.to_csv(f"./test/SCi_FullCV_{str(i)}.csv", index=False)
