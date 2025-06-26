from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import random

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

from chemprop.features import PARAMS

def setup_seed(seed): 
    random.seed(seed)                        
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True

setup_seed(0)
Uxx = torch.rand((1,10000,10000)) #, device=args.device

class LRL(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(LRL, self).__init__()
        self.in_dim = in_dim
        self.hidden = 16
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension.
        :param bias: Whether to add bias to linear layers.
        :param depth: Number of message passing steps.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_only_dim = PARAMS.ATOM_FDIM
        self.bond_only_dim = PARAMS.BOND_FDIM
        self.atom_messages = args.atom_messages
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.args = args
        
        self.mode = 'train'
        self.vflag = True
        self.tflag = True
        
        if (self.args.perm_type == 'soft'):
            self.soft_perm = True #True False None
        if self.args.use_lrl_network:
            self.ntwk = LRL(self.hidden_size,self.bond_fdim)
                
        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        #GS_net
        if self.soft_perm==True:
            self.FF_node = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size*2), self.act_func, nn.Linear(self.hidden_size*2,self.hidden_size))
        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.is_atom_bond_targets:
            self.W_o_b = nn.Linear(self.bond_fdim + self.hidden_size, self.hidden_size)

        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

        if args.bond_descriptors == 'descriptor':
            self.bond_descriptors_size = args.bond_descriptors_size
            self.bond_descriptors_layer = nn.Linear(self.hidden_size + self.bond_descriptors_size,
                                                    self.hidden_size + self.bond_descriptors_size,)
        
        self.regularizers = {
            'zero_reg':
                lambda P, I: torch.zeros_like(torch.norm(P * (1 - I)).sum()),
            'squared_fro_norm_inv':
                lambda P, I: torch.square(torch.norm(P * (1 - I))).sum(),
            'batched_fro_norm':
                lambda P, I: torch.norm(P * I, dim = 0).sum(),
            'batched_fro_norm_inv':
                lambda P, I: torch.norm(P * (1 - I), dim = 0).sum(),
            'squared_fro_norm':
                lambda P, I: torch.square(torch.norm(P * I)).sum(),
            'fro_norm_inv':
                lambda P, I: torch.norm(P * (1 - I), dim=[1,2]).sum(),
            'sqrt_squared_fro_norm_inv':
                lambda P, I: torch.sqrt(torch.square(torch.norm(P * (1 - I))).sum()),
            'abs_norm_inv':
                lambda P, I: torch.abs(P * (1 - I)).sum()
        }
    
    def pytorch_sample_gumbel(self, shape, val_perm = None, eps=1e-20):
        #Sample from Gumbel(0, 1)
        if self.mode == 'train':
            self.perm = list(range(1000))
            random.shuffle(self.perm)
        if self.mode == 'test' and not self.tflag:
            self.perm = list(range(1000))
            random.shuffle(self.perm)
            self.tflag = True
        if self.mode == 'val':
            Ux = Uxx[0,val_perm[:shape[1]],val_perm[:shape[2]]]
        else:
            Ux = Uxx[0,self.perm[:shape[1]],self.perm[:shape[2]]]
        return -torch.log(eps - torch.log(Ux + eps)).to(self.device)
        
        
    def pytorch_sinkhorn_iters(self, log_alpha, val_perm, indicator = None, ptype='node'):
        noise_factor = self.args.gumbel_noise_factor
        temp = self.args.gumbel_temperature
        n_iters = self.args.sinkhorn_iters
        batch_size = log_alpha.size()[0]
        n = log_alpha.size()[1]
        '''
        intra = torch.zeros(batch_size, batch_size).to(device)
        for z, n_pt in points:
            if ptype=='edge':
                n_pt=n_pt//2
                z=z//2 + 1
            ones = torch.ones(n_pt, n_pt).to(device)
            intra[z:(z+n_pt), z:(z+n_pt)] = ones
        '''    
        noise = self.pytorch_sample_gumbel(log_alpha.shape, val_perm)*noise_factor #[-1, n, n]
        log_alpha = log_alpha + noise
        log_alpha = torch.div(log_alpha,temp)
        
        if self.args.use_indicator:
            log_alpha = log_alpha + torch.log(indicator)

        for i in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        
        return torch.squeeze(torch.exp(log_alpha), 0)

    def regularizer(self):
        return self.regularizers[self.args.perm_regularizer]

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None,
                val_perm = None) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        
        soft_perm = self.soft_perm #True False None
        
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)
        
        f_atoms_reac, f_atoms_prod_, f_bonds_reac, f_bonds_prod_, adj_reac, adj_prod_, perm, num_atoms, a2b, b2a, b2revb, a_scope, b_scope, indicator = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms_reac, f_atoms_prod_, f_bonds_reac, f_bonds_prod_, adj_reac, adj_prod_, perm, num_atoms, a2b, b2a, b2revb, indicator = f_atoms_reac.squeeze(1).to(self.device), f_atoms_prod_.squeeze(1).to(self.device), f_bonds_reac.squeeze(1).to(self.device), f_bonds_prod_.squeeze(1).to(self.device), adj_reac.squeeze(1).to(self.device), adj_prod_.squeeze(1).to(self.device), perm.squeeze(1).to(self.device), num_atoms.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device), indicator.to(self.device)
        if soft_perm==True:
            self.f_atoms_reac_sep = torch.cat([f_atoms_reac, f_atoms_reac[:, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            self.f_atoms_prod_sep = torch.cat([f_atoms_prod_, f_atoms_prod_[:, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            self.f_bonds_reac_sep = torch.cat([self.f_atoms_reac_sep.unsqueeze(2).expand(-1, -1, f_bonds_reac.shape[2], -1), f_bonds_reac, f_bonds_reac], dim=-1)
            self.f_bonds_prod_sep = torch.cat([self.f_atoms_prod_sep.unsqueeze(2).expand(-1, -1, f_bonds_prod_.shape[2], -1), f_bonds_prod_, f_bonds_prod_], dim=-1)
            
            input_reac = self.W_i(adj_reac.unsqueeze(-1)*self.f_bonds_reac_sep)
            input_prod = self.W_i(adj_prod_.unsqueeze(-1)*self.f_bonds_prod_sep)
            message_r = self.act_func(input_reac)
            message_p = self.act_func(input_prod)
            
            for depth in range(self.depth - 1):
                a_message_r = adj_reac.unsqueeze(-1)*(message_r.sum(dim=1).unsqueeze(1).expand(-1, message_r.shape[1], -1, -1))
                message_r = a_message_r - message_r #a_message[b2a] - rev_message  # num_bonds x hidden
                message_r = message_r.permute(0, 2, 1, 3)
                
                a_message_p = adj_prod_.unsqueeze(-1)*(message_p.sum(dim=1).unsqueeze(1).expand(-1, message_p.shape[1], -1, -1))
                message_p = a_message_p - message_p #a_message[b2a] - rev_message  # num_bonds x hidden
                message_p = message_p.permute(0, 2, 1, 3)
                
            message_r = self.W_h(message_r)
            message_r = self.act_func(input_reac + message_r)  # num_bonds x hidden_size
            message_r = self.dropout(message_r)
            
            message_p = self.W_h(message_p)
            message_p = self.act_func(input_prod + message_p)  # num_bonds x hidden_size
            message_p = self.dropout(message_p)
            
            a_message_r = message_r.sum(dim=1)
            a_input_r = torch.cat([self.f_atoms_reac_sep, a_message_r], dim=-1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens_r = self.act_func(self.W_o(a_input_r))  # num_atoms x hidden
            atom_hiddens_r = self.dropout(atom_hiddens_r)
            
            a_message_p = message_p.sum(dim=1)
            a_input_p = torch.cat([self.f_atoms_prod_sep, a_message_p], dim=-1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens_p = self.act_func(self.W_o(a_input_p))  # num_atoms x hidden
            atom_hiddens_p = self.dropout(atom_hiddens_p)
            
            permutation = self.pytorch_sinkhorn_iters(torch.einsum('bij,bjd->bid', self.FF_node(atom_hiddens_r), self.FF_node(atom_hiddens_p).mT), val_perm, indicator, ptype='node')
            
            if self.args.use_lrl_network:
                f_message_r = self.ntwk(message_r)
                f_message_p = self.ntwk(message_p)
                adj_reac_f = f_message_r * adj_reac.unsqueeze(-1)
                adj_prod_f_ = f_message_p * adj_prod_.unsqueeze(-1)
            else:
                adj_reac_f = self.f_bonds_reac_sep * adj_reac.unsqueeze(-1)
                adj_prod_f_ = self.f_bonds_prod_sep * adj_prod_.unsqueeze(-1)
            
            adj_reac = message_r * adj_reac.unsqueeze(-1)
            adj_prod_ = message_p * adj_prod_.unsqueeze(-1)

        elif soft_perm==False:
            permutation = perm
        elif soft_perm==None:
            permutation = torch.eye(len(f_atoms_reac[0])).unsqueeze(0).expand(len(f_atoms_reac), -1, -1).to(self.device)
           
        f_atoms_prod = torch.einsum('bij,bjd->bid', permutation, f_atoms_prod_)
        f_bonds_prod = torch.einsum('bikd,bkl->bild', torch.einsum('bij,bjkd->bikd', permutation, f_bonds_prod_), permutation.mT)
        
        if soft_perm==True:
            adj_prod_f = torch.einsum('bikd,bkl->bild', torch.einsum('bij,bjkd->bikd', permutation, adj_prod_f_), permutation.mT)
            adj_prod = torch.einsum('bikd,bkl->bild', torch.einsum('bij,bjkd->bikd', permutation, adj_prod_), permutation.mT)
        else:
            adj_prod = torch.einsum('bik,bkl->bil', torch.einsum('bij,bjk->bik', permutation, adj_prod_), permutation.mT)
        
        if soft_perm==True:
            adj_u_f = torch.maximum(adj_reac_f, adj_prod_f)
        adj_u = torch.maximum(adj_reac, adj_prod)
        
        if self.args.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
            f_atoms_diff = f_atoms_prod - f_atoms_reac
            f_bonds_diff = f_bonds_prod - f_bonds_reac
            
        if self.args.reaction_mode in ['reac_prod', 'reac_prod_balance']:
            self.f_atoms = torch.cat([f_atoms_reac, f_atoms_prod[:, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond = torch.cat([f_bonds_reac, f_bonds_prod], dim=-1)
            
        elif self.args.reaction_mode in ['reac_diff', 'reac_diff_balance']:
            self.f_atoms = torch.cat([f_atoms_reac, f_atoms_diff[:, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond = torch.cat([f_bonds_reac, f_bonds_diff], dim=-1)
            
        elif self.args.reaction_mode in ['prod_diff', 'prod_diff_balance']:
            self.f_atoms = torch.cat([f_atoms_prod, f_atoms_diff[:, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond = torch.cat([f_bonds_prod, f_bonds_diff], dim=-1)
        
        self.f_bonds = torch.cat([self.f_atoms.unsqueeze(2).expand(-1, -1, f_bond.shape[2], -1), f_bond], dim=-1)
        
        if soft_perm==True:
            self.f_bonds = torch.sigmoid(adj_u_f)*self.f_bonds #adj_u_init*self.f_bonds
        else:
            self.f_bonds = adj_u.unsqueeze(-1)*self.f_bonds
        
        '''
        f_atoms_prod = node_perm@f_atoms_prod
        if self.args.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
            f_atoms_diff = torch.tensor([list(map(lambda x, y: x - y, ii, jj)) for ii, jj in zip(f_atoms_prod, f_atoms_reac)]).to(self.device)
        if self.args.reaction_mode in ['reac_prod', 'reac_prod_balance']:
            self.f_atoms = [torch.cat((x,y[PARAMS.MAX_ATOMIC_NUM+1:]), -1).tolist() for x,y in zip(f_atoms_reac, f_atoms_prod)]
        elif self.args.reaction_mode in ['reac_diff', 'reac_diff_balance']:
            self.f_atoms = [torch.cat((x,y[PARAMS.MAX_ATOMIC_NUM+1:]), -1).tolist() for x,y in zip(f_atoms_reac, f_atoms_diff)]
        elif self.args.reaction_mode in ['prod_diff', 'prod_diff_balance']:
            self.f_atoms = [torch.cat((x, y[PARAMS.MAX_ATOMIC_NUM+1:]), -1).tolist() for x,y in zip(f_atoms_prod, f_atoms_diff)]
        
        if self.args.reaction_mode in ['reac_diff', 'prod_diff', 'reac_diff_balance', 'prod_diff_balance']:
            f_bond_diff =  edge_perm@f_bonds_prod - f_bonds_reac
        if self.args.reaction_mode in ['reac_prod', 'reac_prod_balance']:
            f_bond = torch.cat((f_bonds_reac, edge_perm@f_bonds_prod), 1)
        elif self.args.reaction_mode in ['reac_diff', 'reac_diff_balance']:
            f_bond = torch.cat((f_bonds_reac, f_bond_diff), 1)
        elif self.args.reaction_mode in ['prod_diff', 'prod_diff_balance']:
            f_bond = torch.cat((edge_perm@f_bonds_prod, f_bond_diff), 1)
        
        self.f_bonds = []   
        for i, atom in enumerate(b2a):
            if i==0:
                self.f_bonds.append(torch.cat((torch.tensor(self.f_atoms[atom]).to(self.device), f_bond[i//2])).tolist())
            elif i%2!=0:
                self.f_bonds.append(torch.cat((torch.tensor(self.f_atoms[atom]).to(self.device), f_bond[i//2 + 1])).tolist())
            else:
                self.f_bonds.append(torch.cat((torch.tensor(self.f_atoms[atom]).to(self.device), f_bond[i//2])).tolist())
        '''
        if self.is_atom_bond_targets:
            b2br = mol_graph.get_b2br().to(self.device)
            if bond_descriptors_batch is not None:
                forward_index = b2br[:, 0]
                backward_index = b2br[:, 1]
                descriptors_batch = np.concatenate(bond_descriptors_batch, axis=0)
                bond_descriptors_batch = np.zeros([descriptors_batch.shape[0] * 2 + 1, descriptors_batch.shape[1]])
                for i, fi in enumerate(forward_index):
                    bond_descriptors_batch[fi] = descriptors_batch[i]
                for i, fi in enumerate(backward_index):
                    bond_descriptors_batch[fi] = descriptors_batch[i]
                bond_descriptors_batch = torch.from_numpy(bond_descriptors_batch).float().to(self.device)
        
        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(self.f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(self.f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size
        
        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                if soft_perm==True:
                    a_message = adj_u*(message.sum(dim=1).unsqueeze(1).expand(-1, message.shape[1], -1, -1))
                else:
                    a_message = adj_u.unsqueeze(-1)*(message.sum(dim=1).unsqueeze(1).expand(-1, message.shape[1], -1, -1))
                message = a_message - message #a_message[b2a] - rev_message  # num_bonds x hidden
                message = message.permute(0, 2, 1, 3)
            
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x hidden
        
        # atom hidden
        '''
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        '''
        a_message = message.sum(dim=1)
        a_input = torch.cat([self.f_atoms, a_message], dim=-1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden
        
        # bond hidden
        if self.is_atom_bond_targets:
            b_input = torch.cat([f_bonds, message], dim=-1)  # num_bonds x (bond_fdim + hidden)
            bond_hiddens = self.act_func(self.W_o_b(b_input))  # num_bonds x hidden
            bond_hiddens = self.dropout(bond_hiddens)  # num_bonds x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError('The number of atoms is different from the length of the extra atom features')
            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # concatenate the bond descriptors
        if self.is_atom_bond_targets and bond_descriptors_batch is not None:
            if len(bond_hiddens) != len(bond_descriptors_batch):
                raise ValueError('The number of bonds is different from the length of the extra bond features')
            bond_hiddens = torch.cat([bond_hiddens, bond_descriptors_batch], dim=1)     # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.bond_descriptors_layer(bond_hiddens)                    # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.dropout(bond_hiddens)                             # num_bonds x (hidden + descriptor size)

        # Readout
        if self.is_atom_bond_targets:
            return atom_hiddens, a_scope, bond_hiddens, b_scope, b2br  # num_atoms x hidden, remove the first one which is zero padding
        
        '''
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        '''
        if self.aggregation == 'mean':
            mol_vecs = atom_hiddens.sum(dim=1)
            num_atoms = num_atoms.unsqueeze(1)
            mol_vecs = mol_vecs / num_atoms
        
        elif self.aggregation == 'sum':
            mol_vecs = mol_vec.sum(dim=1)
            
        elif self.aggregation == 'norm':
            mol_vecs = mol_vec.sum(dim=1) / self.aggregation_norm
        
        return mol_vecs, self.regularizer()(permutation, indicator)  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    is_reaction=self.reaction if self.reaction is not False else self.reaction_solvent)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages,
                                                    is_reaction=self.reaction if self.reaction is not False else self.reaction_solvent)
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.bond_descriptors = args.bond_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if not self.reaction_solvent:
            if args.mpn_shared:
                self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
            else:
                self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                             for _ in range(args.number_of_molecules)])
        else:
            self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            # Set separate atom_fdim and bond_fdim for solvent molecules
            self.atom_fdim_solvent = get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   is_reaction=False)
            self.bond_fdim_solvent = get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   overwrite_default_bond=args.overwrite_default_bond_features,
                                                   atom_messages=args.atom_messages,
                                                   is_reaction=False)
            self.encoder_solvent = MPNEncoder(args, self.atom_fdim_solvent, self.bond_fdim_solvent,
                                              args.hidden_size_solvent, args.bias_solvent, args.depth_solvent)
    
    def set_mode(self, mode):
        encoders = list(self.encoder.children())
        for encoder in encoders:
            encoder.mode = mode
            encoder.vflag = False
            encoder.tflag = False

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                val_perm: List[int] = None) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif self.bond_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor' or self.bond_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch, bond_descriptors_batch, val_perm) for enc, ba in zip(self.encoder, batch)]
        else:
            if not self.reaction_solvent:
                encodings = [enc(ba, val_perm=val_perm) for enc, ba in zip(self.encoder, batch)]
            else:
                encodings = []
                for ba in batch:
                    if ba.is_reaction:
                        encodings.append(self.encoder(ba,val_perm=val_perm))
                    else:
                        encodings.append(self.encoder_solvent(ba))

        output, perm_loss = encodings[0] if len(encodings) == 1 else torch.cat(encodings, dim=1)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output, perm_loss
