import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import numpy as np

def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_dof_mechanism(df):
    df = df.copy()

    #Proton abstraction
    fwd_template = '[C:1][OH:2].[N:3]=[*:4]>>[C:1][O-:2].[NH+:3]=[*:4]'
    rxn1 = rdChemReactions.ReactionFromSmarts(fwd_template)
    oxide = []
    conj_a = []

    #Intermediate formation
    fwd_template = '[C:1][O-:2].[S:3][F:4]>>[C:1][O+0:2][S:3]'
    rxn2 = rdChemReactions.ReactionFromSmarts(fwd_template)
    sulfoxide = []

    #Substitution
    fwd_template1 = '[C@H:1][O:2][S:3]>>[C@@H:1][F:4]'
    fwd_template2 = '[C:1][O:2][S:3]>>[C:1][F:4]'
    rxn3 = rdChemReactions.ReactionFromSmarts(fwd_template1)
    rxn4 = rdChemReactions.ReactionFromSmarts(fwd_template2)
    pdt = []

    for i, row in df.iterrows():
        #print('Reaction no.: ', i)
        #Proton abstraction
        reacts1 = (Chem.MolFromSmiles(row['alcohol']), Chem.MolFromSmiles(row['base']))
        abstracts = rxn1.RunReactants(reacts1)
        oxide_smiles = set([Chem.MolToSmiles(mol[0]) for mol in abstracts])
        if len(list(oxide_smiles)) > 1:
            oxide.append(list(oxide_smiles)[1])
        else:
            oxide.append(list(oxide_smiles)[0])
        conj_a_smiles = set([Chem.MolToSmiles(mol[1]) for mol in abstracts])
        conj_a.append(list(conj_a_smiles)[0])
        #print("No. of possible conjugated acid: ", len(list(conj_a_smiles)))
        #print("No. of possible oxide: ", len(list(oxide_smiles)))
       
        #Intermediate formation
        reacts2 = (Chem.MolFromSmiles(list(oxide_smiles)[0]), Chem.MolFromSmiles(row['sulfonyl_fluoride']))
        anchors = rxn2.RunReactants(reacts2)
        sulfoxide_smiles = set([Chem.MolToSmiles(mol[0]) for mol in anchors])
        #assert len(sulfoxide_smiles) == 1
        sulfoxide.append(list(sulfoxide_smiles)[0])
        #print("No. of possible sulfoxide: ", len(list(sulfoxide_smiles)))

        #Substitution
        try:
            reacts3 = (Chem.MolFromSmiles(list(sulfoxide_smiles)[0]),)
            attacks = rxn3.RunReactants(reacts3)
            pdt_smiles = set([Chem.MolToSmiles(mol[0]) for mol in attacks])
            #assert len(pdt_smiles) == 1
            pdt.append(list(pdt_smiles)[0])
            #print("No. of possible product: ", len(list(pdt_smiles)))
        except IndexError:
            reacts3 = (Chem.MolFromSmiles(list(sulfoxide_smiles)[0]),)
            attacks = rxn4.RunReactants(reacts3)
            pdt_smiles = set([Chem.MolToSmiles(mol[0]) for mol in attacks])
            #assert len(pdt_smiles) == 1
            pdt.append(list(pdt_smiles)[0])
            #print("No. of possible product: ", len(list(pdt_smiles)))

        ##multiple oxides
        #if len(list(oxide_smiles)) > 1:
            #print(f'Reaction {i} has oxide {len(list(oxide_smiles))}')
            #print(list(oxide_smiles))
    df['oxide'] = oxide
    df['conjugate_acid'] = conj_a
    df['sulfoxide'] = sulfoxide
    df['product'] = pdt
    F = Chem.MolToSmiles(Chem.MolFromSmiles('[F-]'))
    df['fluoride'] = F
    
    return df
    
for i in range(1, 11):
    sheet_id = str(i).zfill(2)
    dfs = pd.read_excel("DFS75_data.xlsx", sheet_name='FullCV_'+sheet_id)
    df_ = generate_dof_mechanism(dfs)
    rxns = []
    for j, row in df_.iterrows():
        rxn = f"{row['alcohol']}*{row['base']}*{row['sulfonyl_fluoride']}*{row['product']}"
        rxns.append(rxn)
        
    df_['reaction'] = rxns
    df_['temp'] = ''
    df_['output'] = df_['Output']
    df__ = df_[['reaction', 'temp', 'output']]
    df = df__.drop_duplicates()
    train, val, test = df.iloc[:140], df.iloc[140:160], df.iloc[160:]
    df.to_csv(f"./full/DF_FullCV_{str(i)}.csv", index=False)
    train.to_csv(f"./train/DF_FullCV_{str(i)}.csv", index=False)
    val.to_csv(f"./val/DF_FullCV_{str(i)}.csv", index=False)
    test.to_csv(f"./test/DF_FullCV_{str(i)}.csv", index=False)


