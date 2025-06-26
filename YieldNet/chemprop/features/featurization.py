from typing import List, Tuple, Union
from itertools import zip_longest
import logging

from rdkit import Chem
import torch
import random
import numpy as np

from chemprop.rdkit import make_mol

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
        self.ADDING_H = False
        self.KEEP_ATOM_MAP = False
        
        self.R2P_SEED = 20
# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()

def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(overwrite_default_atom: bool = False, is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors.
    :param is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None.
    :return: The dimensionality of the atom feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + is_reaction * PARAMS.EXTRA_ATOM_FDIM
    else:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h

def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.

    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs

def set_keeping_atom_map(keeping_atom_map: bool) -> None:
    """
    Sets whether RDKit molecules keep the original atom mapping.

    :param keeping_atom_map: Boolean whether to keep the original atom mapping.
    """
    PARAMS.KEEP_ATOM_MAP = keeping_atom_map

def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.
 
    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode
        
def is_explicit_h(is_mol: bool = True) -> bool:
    r"""Returns whether to retain explicit Hs (for reactions only)"""
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False


def is_adding_hs(is_mol: bool = True) -> bool:
    r"""Returns whether to add explicit Hs to the mol (not for reactions)"""
    if is_mol:
        return PARAMS.ADDING_H
    return False


def is_keeping_atom_map(is_mol: bool = True) -> bool:
    r"""Returns whether to keep the original atom mapping (not for reactions)"""
    if is_mol:
        return PARAMS.KEEP_ATOM_MAP
    return True


def is_reaction(is_mol: bool = True) -> bool:
    r"""Returns whether to use reactions as input"""
    if is_mol:
        return False
    if PARAMS.REACTION: #(and not is_mol, checked above)
        return True
    return False


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False,
                  is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors.
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors.
    :param is_reaction: Whether to add :code:`EXTRA_BOND_FDIM` for reaction input when :code:`REACTION_MODE:` is not None
    :return: The dimensionality of the bond feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + is_reaction * PARAMS.EXTRA_BOND_FDIM + \
            (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction)
    else:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM + \
            (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
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


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
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


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1) #set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
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


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()]) 
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx()) 
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    pi_values =  list(reac_id_to_prod_id.values())
    random.seed(PARAMS.R2P_SEED)
    random.shuffle(pi_values)
    # reassigning to keys
    res = dict(zip(reac_id_to_prod_id, pi_values)) #reac_id_to_prod_id
    return res, only_prod_ids, only_reac_ids #res

def permutation_mat(n, r2p, ro, po):
    perm = np.zeros((n, n))
    for items in r2p.items():
        perm[items] = 1.0
    if len(ro)!=0:
        for i, index in enumerate(ro):
            perm[index, len(r2p.items())+len(po)+i] = 1.0
    if len(po)!=0:
        for i, index in enumerate(po):
            perm[index, len(r2p.items())+i] = 1.0
    return perm

class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    * :code:`is_mol`: A boolean whether the input is a molecule.
    * :code:`is_reaction`: A boolean whether the molecule is a reaction.
    * :code:`is_explicit_h`: A boolean whether to retain explicit Hs (for reaction mode).
    * :code:`is_adding_hs`: A boolean whether to add explicit Hs (not for reaction mode).
    * :code:`reaction_mode`:  Reaction mode to construct atom and bond feature vectors.
    * :code:`b2br`: A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule.
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating.
        """
        self.is_mol = is_mol(mol)
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)
        self.is_keeping_atom_map = is_keeping_atom_map(self.is_mol)
        self.reaction_mode = reaction_mode()
        
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map), make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map)) 
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs, self.is_keeping_atom_map)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        if not self.is_reaction:
            # Get atom features
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Initialize f_bonds to real bonds mapping for each bond
            self.b2br = np.zeros([len(mol.GetBonds()), 2])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.b2br[bond.GetIdx(), :] = [self.n_bonds, self.n_bonds + 1]
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')

        else: # Reaction mode
            if atom_features_extra is not None:
                raise NotImplementedError('Extra atom features are currently not supported for reactions')
            if bond_features_extra is not None:
                raise NotImplementedError('Extra bond features are currently not supported for reactions')

            mol_reac = mol[0]
            mol_prod = mol[1]
            ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
            
            # Get atom features
            if self.reaction_mode in ['reac_diff','prod_diff', 'reac_prod']:
                #Reactant: regular atom features for each atom in the reactants, as well as zero features for atoms that are only in the products (indices in pio)
                self.f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio]
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), other atom features zero,
                #regular features for atoms that are only in the products (indices in pio)
                self.f_atoms_prod = [atom_features(atom) for atom in mol_prod.GetAtoms()] + [atom_features_zeros(mol_reac.GetAtomWithIdx(index)) for index in rio]
            
            else: #balance
                #Reactant: regular atom features for each atom in the reactants, copy features from product side for atoms that are only in the products (indices in pio)
                self.f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
                
                #Product: regular atom features for each atom that is in both reactants and products (not in rio), copy features from reactant side for
                #other atoms, regular features for atoms that are only in the products (indices in pio)
                self.f_atoms_prod = [atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()])) if atom.GetIdx() not in rio else
                                atom_features(atom) for atom in mol_reac.GetAtoms()] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
            
            self.perm_mat = permutation_mat(len(self.f_atoms_reac), ri2pi, rio, pio) #, random_perm=True
            self.n_atoms = len(self.f_atoms_reac)
            n_atoms_reac = mol_reac.GetNumAtoms()

            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])
            
            self.adj_r = Chem.rdmolops.GetAdjacencyMatrix(mol_reac)
            self.adj_p = Chem.rdmolops.GetAdjacencyMatrix(mol_prod)
            self.r_edge_feature = np.zeros((self.n_atoms, self.n_atoms, PARAMS.BOND_FDIM))
            self.p_edge_feature = np.zeros((self.n_atoms, self.n_atoms, PARAMS.BOND_FDIM))
            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                    bond_prod = mol_prod.GetBondBetweenAtoms(a1, a2)
                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    self.r_edge_feature[a1, a2, :] = f_bond_reac
                    self.r_edge_feature[a2, a1, :] = f_bond_reac
                    self.p_edge_feature[a1, a2, :] = f_bond_prod
                    self.p_edge_feature[a2, a1, :] = f_bond_prod


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    * :code:`b2br`: (Optional): A mapping from f_bonds to real bonds in molecule recorded in targets.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.mol_graphs = mol_graphs
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features,
                                       is_reaction=self.is_reaction)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                      overwrite_default_atom=self.overwrite_default_atom_features,
                                      is_reaction=self.is_reaction)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        self.atom_only_dim = len(mol_graphs[0].f_atoms_reac[0])
        self.bond_only_dim = len(mol_graphs[0].r_edge_feature[0])
        
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]
        
        self.max_num_atoms = max(1, max(len(mol_graph.f_atoms_reac) for mol_graph in mol_graphs))
        f_atoms_reac = []
        f_atoms_prod = []
        f_bonds_reac = []
        f_bonds_prod = []
        adj_reac = []
        adj_prod = []
        perm = []
        num_atoms = []
        for mol_graph in mol_graphs:
            num_atoms.append(len(mol_graph.f_atoms_reac))
            edge_r = np.zeros((self.max_num_atoms, self.max_num_atoms, PARAMS.BOND_FDIM))
            edge_p = np.zeros((self.max_num_atoms, self.max_num_atoms, PARAMS.BOND_FDIM))
            A_r = np.zeros((self.max_num_atoms, self.max_num_atoms))
            A_p = np.zeros((self.max_num_atoms, self.max_num_atoms))
            
            P = np.eye(self.max_num_atoms)
            f_r = mol_graph.f_atoms_reac + [[0] * PARAMS.ATOM_FDIM] * (self.max_num_atoms - len(mol_graph.f_atoms_reac))
            f_p = mol_graph.f_atoms_prod + [[0] * PARAMS.ATOM_FDIM] * (self.max_num_atoms - len(mol_graph.f_atoms_prod))
            f_atoms_reac.append(f_r)
            f_atoms_prod.append(f_p)
            
            edge_r[:len(mol_graph.r_edge_feature), :len(mol_graph.r_edge_feature)] = mol_graph.r_edge_feature
            edge_p[:len(mol_graph.p_edge_feature), :len(mol_graph.p_edge_feature)] = mol_graph.p_edge_feature
            f_bonds_reac.append(edge_r)
            f_bonds_prod.append(edge_p)
            
            A_r[:len(mol_graph.adj_r), :len(mol_graph.adj_r)] = mol_graph.adj_r
            A_p[:len(mol_graph.adj_p), :len(mol_graph.adj_p)] = mol_graph.adj_p
            adj_reac.append(A_r)
            adj_prod.append(A_p)
            
            P[:len(mol_graph.perm_mat), :len(mol_graph.perm_mat)] = mol_graph.perm_mat
            perm.append(P)

        self.f_atoms_reac = torch.tensor(f_atoms_reac, dtype=torch.float32)
        self.f_atoms_prod = torch.tensor(f_atoms_prod, dtype=torch.float32)
        self.f_bonds_reac = torch.tensor(np.array(f_bonds_reac), dtype=torch.float32)
        self.f_bonds_prod = torch.tensor(np.array(f_bonds_prod), dtype=torch.float32)

        self.adj_reac = torch.tensor(np.array(adj_reac), dtype=torch.float32)
        self.adj_prod = torch.tensor(np.array(adj_prod), dtype=torch.float32)
        self.perm = torch.tensor(np.array(perm), dtype=torch.float32)
        self.num_atoms =  torch.tensor(num_atoms)
        
        B, N, _ = self.f_atoms_reac.shape
        self.indicator = torch.ones((B,N,N)) - torch.floor(torch.cdist(self.f_atoms_reac[:,:,:100],self.f_atoms_prod[:,:,:100],p=4))
        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.a2b = torch.tensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)], dtype=torch.long)
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.b2br = None  # only needed in predictions of atomic/bond targets

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                                   torch.Tensor, torch.Tensor, torch.Tensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds_reac = self.f_bonds_reac
            f_bonds_prod = self.f_bonds_prod 

        return self.f_atoms_reac, self.f_atoms_prod, self.f_bonds_reac, self.f_bonds_prod, self.adj_reac, self.adj_prod, self.perm, self.num_atoms, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.indicator

    def get_b2b(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

    def get_b2br(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from f_bonds to real bonds in molecule recorded in targets.

        :return: A PyTorch tensor containing the mapping from f_bonds to real bonds in molecule recorded in targets.
        """
        if self.b2br is None:
            n_bonds = 1 # number of bonds (start at 1 b/c need index 0 as padding)
            b2br = []
            for mol_graph in self.mol_graphs:
                b2br.append(mol_graph.b2br + n_bonds)
                n_bonds += mol_graph.n_bonds
            b2br = np.concatenate(b2br, axis=0)
            self.b2br = torch.tensor(b2br, dtype=torch.long)

        return self.b2br

def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule.
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule.
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating.
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating.
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf
                          in zip_longest(mols, atom_features_batch, bond_features_batch)])

def is_mol(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    """Checks whether an input is a molecule or a reaction

    :param mol: str, RDKIT molecule or tuple of molecules.
    :return: Whether the supplied input corresponds to a single molecule.
    """

    if isinstance(mol, str) and ">" not in mol:
        return True
    elif isinstance(mol, Chem.Mol):
        return True
    return False
