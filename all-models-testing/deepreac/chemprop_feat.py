import rdkit 
from rdkit import Chem
from dgllife.utils import BaseAtomFeaturizer, ConcatFeaturizer

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

def one_hot_encoding(x, allowable_set, encode_unknown=True):
    
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))


def atomic_number_one_hot(atom, allowable_set=None):

    if allowable_set is None:
        allowable_set = list(range(100))
    return one_hot_encoding(atom.GetAtomicNum()-1, allowable_set)
    
def atom_degree_one_hot(atom, allowable_set=None):
    
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetDegree(), allowable_set)
    
def atom_formal_charge_one_hot(atom, allowable_set=None):

    if allowable_set is None:
        allowable_set = [-1, -2, 1, 2, 0]
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set)
    
def atom_chiral_one_hot(atom, allowable_set=None):

    if allowable_set is None:
        allowable_set = [0, 1, 2, 3]
    return one_hot_encoding(atom.GetChiralTag(), allowable_set)

def atom_total_num_H_one_hot(atom, allowable_set=None):
    
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set)
    
def atom_hybridization_one_hot(atom, allowable_set=None):
    
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set)
    
def atom_is_aromatic(atom):
    
    if atom.GetIsAromatic():
        return [True]
        
    else:
        return [False]
        
def atomic_mass(atom):
    
    return [atom.GetMass() * 0.01]

class ChempropFeaturizer(BaseAtomFeaturizer):

    def __init__(self, atom_data_field='h'):
        super(ChempropFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atomic_number_one_hot,
                 atom_degree_one_hot,
                 atom_formal_charge_one_hot,
                 atom_chiral_one_hot,
                 atom_total_num_H_one_hot,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atomic_mass]
            )})

    