import os, csv
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

def mol_dict():
    return {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': []}


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.MULTISTEP = False ###changed
        self.ADDING_H = False
        self.KEEP_ATOM_MAP = False
        
        self.R2P_SEED = 20
# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()

def onek_encoding_unk(value: int, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups= None):
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features
    
def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def get_graph_data(rsmi_list, yld_list, data_id, filename):

    def add_mol(mol_dict, mol):
        x = [a.GetSymbol() for a in mol.GetAtoms()]
        l = [str(a.GetHybridization()) for a in mol.GetAtoms()]
        # for i in range(0,len(l)):
        #     if(l[i]=='UNSPECIFIED'):
        #         print(i, x[i])

        def _DA(mol):
    
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list

        def _chirality(atom):
            
            if atom.HasProp('Chirality'):
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
            else:
                c_list = [0, 0]

            return c_list
            
        def _stereochemistry(bond):
            
            if bond.HasProp('Stereochemistry'):
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
            else:
                s_list = [0, 0]

            return s_list     
            

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        '''
        D_list, A_list = _DA(mol) 
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        '''
        node_attr = np.array([atom_features(atom) for atom in mol.GetAtoms()])
        #node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])
        
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
    
        if n_edge > 0:
            '''
            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            '''
            edge_attr = np.array([bond_features(b) for b in mol.GetBonds()])
            #edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
            edge_attr = np.vstack([edge_attr, edge_attr])
    
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
            mol_dict['edge_attr'].append(edge_attr)
            mol_dict['src'].append(src)
            mol_dict['dst'].append(dst)
        
        return mol_dict

    def add_dummy(mol_dict):

        n_node = 1
        n_edge = 0
        node_attr = np.zeros((1, PARAMS.ATOM_FDIM))
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
        
        return mol_dict
  
    def dict_list_to_numpy(mol_dict):
    
        mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
        mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
        mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
        if np.sum(mol_dict['n_edge']) > 0:
            mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
            mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
            mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
        else:
            mol_dict['edge_attr'] = np.empty((0, PARAMS.BOND_FDIM)).astype(bool)
            mol_dict['src'] = np.empty(0).astype(int)
            mol_dict['dst'] = np.empty(0).astype(int)

        return mol_dict
    
    rmol_max_cnt = np.max([smi.split('>>')[0].count('.') + 1 for smi in rsmi_list])
    imol_max_cnt = np.max([smi.split('>>')[1].count('.') + 1 for smi in rsmi_list])
    pmol_max_cnt = np.max([smi.split('>>')[2].count('.') + 1 for smi in rsmi_list])
    # pmol_max_cnt = np.max([smi.split('>>')[1].count('.') + 1 for smi in rsmi_list])

    rmol_dict = [mol_dict() for _ in range(rmol_max_cnt)]
    imol_dict = [mol_dict() for _ in range(imol_max_cnt)]
    pmol_dict = [mol_dict() for _ in range(pmol_max_cnt)]
                 
    reaction_dict = {'yld': [], 'rsmi': []}     
    
    print('--- generating graph data for %s' %filename)
    print('--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d, intermediate_max_cnt: %d' %(len(rsmi_list), rmol_max_cnt, pmol_max_cnt, imol_max_cnt)) 
                 
    for i in range(len(rsmi_list)):
    
        rsmi = rsmi_list[i].replace('~', '-')
        yld = yld_list[i]
    
        # [reactants_smi, products_smi] = rsmi.split('>>')
        [reactants_smi, intermediate_smi, products_smi] = rsmi.split('>>')
        
        # processing reactants
        reactants_smi_list = reactants_smi.split('.')
        for _ in range(rmol_max_cnt - len(reactants_smi_list)): reactants_smi_list.append('')
        for j, smi in enumerate(reactants_smi_list):
            if smi == '':
                rmol_dict[j] = add_dummy(rmol_dict[j]) 
            else:
                rmol = Chem.MolFromSmiles(smi)
                '''
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                '''
                rmol = Chem.AddHs(rmol)
                #rmol = Chem.RemoveHs(rmol)
                rmol_dict[j] = add_mol(rmol_dict[j], rmol)

        # processing intermediates
        intermediate_smi_list = intermediate_smi.split('.')
        for _ in range(imol_max_cnt - len(intermediate_smi_list)): intermediate_smi_list.append('')
        for j, smi in enumerate(intermediate_smi_list):
            if smi == '':
                imol_dict[j] = add_dummy(imol_dict[j]) 
            else:
                imol = Chem.MolFromSmiles(smi)
                '''
                rs = Chem.FindPotentialStereo(imol)
                for element in rs:
                    # print(str(element.type), str(element.specified), str(element.specified), element.centeredOn)
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': imol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': imol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                '''
                imol = Chem.AddHs(imol)
                #imol = Chem.RemoveHs(imol)
                imol_dict[j] = add_mol(imol_dict[j], imol)
        
        # processing products
        products_smi_list = products_smi.split('.')
        for _ in range(pmol_max_cnt - len(products_smi_list)): products_smi_list.append('') 
        for j, smi in enumerate(products_smi_list):
            if smi == '':
                pmol_dict[j] = add_dummy(pmol_dict[j])
            else: 
                pmol = Chem.MolFromSmiles(smi)
                '''
                ps = Chem.FindPotentialStereo(pmol)
                for element in ps:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                '''
                pmol = Chem.AddHs(pmol)      
                #pmol = Chem.RemoveHs(pmol) 
                pmol_dict[j] = add_mol(pmol_dict[j], pmol)  
        
        # yield and reaction SMILES
        reaction_dict['yld'].append(yld)
        reaction_dict['rsmi'].append(rsmi)
    
        # monitoring
        if (i+1) % 1000 == 0: print('--- %d/%d processed' %(i+1, len(rsmi_list))) 
        
    # datatype to numpy
    for j in range(rmol_max_cnt): rmol_dict[j] = dict_list_to_numpy(rmol_dict[j])   
    for j in range(imol_max_cnt): imol_dict[j] = dict_list_to_numpy(imol_dict[j])   
    for j in range(pmol_max_cnt): pmol_dict[j] = dict_list_to_numpy(pmol_dict[j])   
    reaction_dict['yld'] = np.array(reaction_dict['yld'])
    print(len(imol_dict[1]))
    # save file
    np.savez_compressed(filename, data = [rmol_dict, imol_dict, pmol_dict, reaction_dict]) 
    # np.savez_compressed(filename, data = [rmol_dict, pmol_dict, reaction_dict]) 

    
if __name__ == "__main__":

    for data_id in [0]:
        # for sub in range(1, 3):
        for split_id in range(10):

            load_dict = np.load("./GPT/gp0/data_gp0_%d_total_split_%d.npz" %(data_id, split_id), allow_pickle = True)
            #load_dict = np.load('./split/data%d_split_%d.npz' %(data_id, split_id), allow_pickle = True)
            #load_dict = np.load(f'./split/dof_oob_{split_id}.npz', allow_pickle = True)
            
            rsmi_list = load_dict['data_df'][:,0]
            yld_list = load_dict['data_df'][:,1]
            #filename = './dataset_%d_%d_chemprop.npz' %(data_id, split_id)
            filename = '../../yield_prediction/data/chemprop/gp0/dataset_%d_%d.npz' %(data_id, split_id)
            #filename = './dof_oob_%d.npz' %(split_id) #data_id, 
            
            get_graph_data(rsmi_list, yld_list, data_id, filename)

    # for test_id in [1, 2, 3, 4]:

    #     data_id = 1

    #     load_dict = np.load('./split/Test%d_split.npz' %test_id, allow_pickle = True)

    #     rsmi_list = load_dict['data_df'][:,0]
    #     yld_list = load_dict['data_df'][:,1]
    #     filename = './test_%d.npz' %test_id
    
    #     get_graph_data(rsmi_list, yld_list, data_id, filename)
