from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import random

from torch.nn import functional as F
import dgl
import dgl.function as fn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function, get_regularizer_function

import warnings
warnings.filterwarnings('ignore')
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


def squash(s, dim=1):
    sq = torch.sum(s ** 2, dim=dim, keepdim=True)
    s_norm = torch.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_norm)
    return s

class CapsuleLayer(nn.Module):
    def __init__(self, in_nodes_dim=8, in_nodes=4, out_nodes=1, out_nodes_dim=16, device='cuda'):
        super(CapsuleLayer, self).__init__()
        self.device = device
        self.in_nodes_dim, self.out_nodes_dim = in_nodes_dim, out_nodes_dim
        self.in_nodes, self.out_nodes = in_nodes, out_nodes
        self.weight = nn.Parameter(torch.randn(in_nodes, out_nodes, out_nodes_dim, in_nodes_dim))

    def forward(self, x):
        self.batch_size = x.size(0)
        u_hat = self.compute_uhat(x)
        routing = RoutingLayer(self.in_nodes, self.out_nodes, self.out_nodes_dim, batch_size=self.batch_size,
                                  device=self.device)
        routing(u_hat, routing_num=3)
        out_nodes_feature = routing.g.nodes[routing.out_indx].data['v']
        return out_nodes_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

    def compute_uhat(self, x):
        
        x = torch.stack([x] * self.out_nodes, dim=2).unsqueeze(4)
        W = self.weight.expand(self.batch_size, *self.weight.size())
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()
        return u_hat.view(-1, self.batch_size, self.out_nodes_dim)

class RoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, batch_size=0, device='cuda'):
        super(RoutingLayer, self).__init__()
        self.batch_size = batch_size
        self.g = init_graph(in_nodes, out_nodes, f_size, device=device)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
        self.device = device

    def forward(self, u_hat, routing_num=1):
        self.g.edata['u_hat'] = u_hat
        # batch_size = self.batch_size
        
        # self.g.update_all(cap_message, cap_reduce)

        for r in range(routing_num):
            edges_b = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            
            self.g.edata['c'] = F.softmax(edges_b, dim=1).view(-1, 1)
            
            if self.batch_size:
                self.g.edata['c u_hat'] = self.g.edata['c'].unsqueeze(1) * self.g.edata['u_hat']
            else:
                self.g.edata['c u_hat'] = self.g.edata['c'] * self.g.edata['u_hat']

            self.g.update_all(fn.copy_e('c u_hat', 'm'), fn.sum('m', 's'))

            # self.g.update_all()

            if self.batch_size:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
            else:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)
          
            v = torch.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            
            if self.batch_size:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
            else:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)
            
def init_graph(in_nodes, out_nodes, f_size, device='cuda'):
    g = dgl.DGLGraph()
    g.set_n_initializer(dgl.frame.zero_initializer)
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)
    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)
    g = g.to(device)
    g.edata['b'] = torch.zeros(in_nodes * out_nodes, 1).to(device)
    return g
    

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

class Attention(nn.Module):
    
    def __init__(self, in_dim, out_dim, seq_len):
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.NN0 = nn.Linear(self.in_dim*2, self.in_dim)
        self.NN0_self = nn.Linear(self.in_dim, self.in_dim)
        self.NN1 = nn.Linear(self.in_dim*self.seq_len, self.out_dim)
        
    def forward(self, h):
        
        output = []
        for s1 in range(self.seq_len):
            out_i = []
            for s2 in range(self.seq_len):
                if s1==s2:
                    out_i.append(self.NN0_self(h[:, s1]))
                else:
                    out_i.append(self.NN0(torch.cat([h[:, s1], h[:, s2]], dim=-1)))
            output.append(self.NN1(torch.cat(out_i, dim=-1)).unsqueeze(1))
            
        return torch.cat(output, dim=1)
        
class Set2Set(nn.Module):
    
    def __init__(self, in_dim, device, n_layers=1, n_iters=3):
        super(Set2Set, self).__init__()
        self.in_dim = in_dim
        self.out_dim = 2*in_dim
        self.device = device
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.out_dim, self.in_dim, self.n_layers, batch_first=True)
        
    def forward(self, feat):
        batch_sz = feat.shape[0]
        
        l = (torch.zeros(self.n_layers, batch_sz, feat.shape[-1]).to(self.device),
            torch.zeros(self.n_layers, batch_sz, feat.shape[-1]).to(self.device))
            
        q_star = torch.zeros(batch_sz, self.out_dim).to(self.device)
        
        for _ in range(self.n_iters):
            q, l = self.lstm(q_star.unsqueeze(1), l)
            q = q.view(batch_sz, self.in_dim)
            alpha = torch.softmax((q.unsqueeze(1)*feat).sum(-1, keepdim=True), dim=1)
            r = feat*alpha
            readout = r.sum(1)
            q_star = torch.cat([q, readout], dim=-1)
            
        return q_star

class GNN(nn.Module):

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        
        super(GNN, self).__init__()
        
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.device = args.device
        self.args = args
    
        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)
        
        # Cached zeros
        # self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        
        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        input_dim_bond = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim_bond, self.hidden_size, bias=self.bias)
        
        w_h_input_size = self.hidden_size
        
        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
            
        #self.W_c = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        self.W_o = nn.Linear(self.hidden_size*3, self.hidden_size)

        
    def forward(self, f_atoms, f_bonds, adj, adj_f=None):
    
        if adj_f==None:
            adj_f = adj
            
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        if adj_f.shape==f_bonds.shape:
            input_bond = self.W_i_bond(adj_f*f_bonds)  # num_bonds x hidden_size
        else:
            input_bond = self.W_i_bond(adj_f.unsqueeze(-1)*f_bonds)  # num_bonds x hidden_size
            
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        
        for depth in range(self.depth - 1):
            agg_message = message_bond.sum(dim=2)*torch.sigmoid(message_bond.max(dim=2)[0])
            #message_atom = self.W_c(torch.cat([message_atom, agg_message], -1))
            #resonance = torch.einsum('bsnid,bsndj->bsnij', agg_message.unsqueeze(3), message_atom.unsqueeze(-1)).squeeze(-1)
            message_atom = message_atom + agg_message#*torch.sigmoid(resonance)
            if adj.shape == message_bond.shape:
                resonance = torch.einsum('bsnd,bsmd->bsnm', message_atom, message_atom).unsqueeze(-1)*adj
                message_bond = adj*(message_atom.unsqueeze(2).expand(-1, -1, message_bond.shape[2], -1, -1)) - message_bond
            else:
                resonance = torch.einsum('bsnd,bsmd->bsnm', message_atom, message_atom)*adj
                message_bond = adj.unsqueeze(-1)*(message_atom.unsqueeze(2).expand(-1, -1, message_bond.shape[2], -1, -1)) - message_bond
            message_bond = message_bond.permute(0, 1, 3, 2, 4)
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.act_func(input_bond + message_bond)  # num_bonds x hidden_size
            message_bond = self.dropout(message_bond)  # num_bonds x hidden
            if adj.shape == message_bond.shape:
                message_bond = message_bond*torch.softmax(resonance, dim=2)
            else:
                message_bond = message_bond*torch.softmax(resonance, dim=2).unsqueeze(-1)
                
        agg_message = message_bond.sum(dim=2)*torch.sigmoid(message_bond.max(dim=2)[0])
        a_input = torch.cat([agg_message, message_atom, input_atom], dim=-1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)
        
        return atom_hiddens, message_bond
        
        

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
        self.step_aggregation =  args.step_aggregation
        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.args = args
        self.use_node_tf = args.use_node_tf
        self.use_step_tf = args.use_step_tf
        self.step_tf = args.step_tf
        self.num_of_steps = args.num_of_steps
        #self.num_gru_layers = 4
        self.mode = 'train'
        self.vflag = True
        self.tflag = True
        
        if (self.args.perm_type == 'soft'):
            self.soft_perm = True #True False None
        elif (self.args.perm_type == 'hard'):
            self.soft_perm = False #True False None
        else:
            self.soft_perm = None #True False None
        if self.args.use_lrl_network:
            self.ntwk = LRL(self.hidden_size,self.bond_fdim)
                
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
        # Activation
        self.act_func = get_activation_function(args.activation)
        
        #Regularizer
        self.regularizer = get_regularizer_function(args.perm_regularizer)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        #GS_net
        if self.soft_perm==True:
            self.FF_node = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size*2), self.act_func, nn.Linear(self.hidden_size*2, self.hidden_size))
            #self.FF_node = nn.Linear(self.hidden_size, self.hidden_size-100)

        self.use_sep_gnn = False
        self.use_same_feat = True
        if self.use_sep_gnn:
            if self.use_same_feat:
                self.rpgnn = GNN(args, self.atom_fdim, self.bond_fdim)
                
            else:
                self.rpgnn = GNN(args, self.atom_only_dim, self.bond_only_dim)
            self.tgnn = GNN(args, self.atom_fdim, self.bond_fdim)
        else:
            self.rpgnn = GNN(args, self.atom_fdim, self.bond_fdim)
        
        if self.use_node_tf:   
            self.reducer = nn.Linear(self.hidden_size, self.hidden_size-self.num_of_steps)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.hidden_size, nhead = 5, batch_first=True, norm_first=True).to(self.device)  #, batch_first=True, , dim_feedforward=self.hidden_size
            #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6) #

        if self.use_step_tf:
            if self.step_tf == 'NNAttention':
                self.encoder_layer = Attention(self.hidden_size, self.hidden_size, self.num_of_steps)
            elif self.step_tf == 'Transformer':
                self.reducer = nn.Linear(self.hidden_size, self.hidden_size-self.num_of_steps)
                self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.hidden_size, nhead = 5, batch_first=True, norm_first=True).to(self.device)  #, batch_first=True
                #self.encoder_layer = nn.GRU(self.hidden_size, self.hidden_size, self.num_gru_layers, batch_first=True).to(self.device)
                #self.encoder_layer =nn.MultiheadAttention(self.hidden_size, 6, batch_first=True).to(self.device)
                #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
                #self.final = nn.Sequential(nn.Linear(self.hidden_size*3, self.hidden_size), self.act_func)
                #self.final = nn.Linear(3, 1)
        
        if self.args.perm_type=='nopm':
            self.compactNN = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size*2), self.act_func, nn.Linear(self.hidden_size*2, self.hidden_size))
            
        if self.aggregation == 'set2set':
            self.node_readout = Set2Set(self.hidden_size, device=self.device)
            self.node_condenser = nn.Linear(self.hidden_size*2, self.hidden_size)
            
        if self.step_aggregation == 'set2set':
            self.graph_readout = Set2Set(self.hidden_size, device=self.device)
            self.graph_condenser = nn.Linear(self.hidden_size*2, self.hidden_size)
            
        if self.step_aggregation == 'capsule':
            self.digits = CapsuleLayer(in_nodes_dim=self.hidden_size, in_nodes=self.num_of_steps, out_nodes=1, out_nodes_dim=self.hidden_size, device=self.device)
      
      
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
          
        noise = self.pytorch_sample_gumbel(log_alpha.shape, val_perm)*noise_factor #[-1, n, n]
        log_alpha = log_alpha + noise
        log_alpha = torch.div(log_alpha,temp)
        if self.args.use_indicator:
            log_alpha = log_alpha + torch.log(indicator)

        for i in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        #return torch.squeeze(torch.exp(log_alpha), 0)
        return torch.exp(log_alpha)

    def atom_bond_feat(self, f_atoms_reac, f_bonds_reac, f_atoms_prod=None, f_bonds_prod=None):
        
        if self.args.reaction_mode in ['reac_diff', 'prod_diff', 'reac_prod', 'reac_diff_balance', 'prod_diff_balance', 'reac_prod_balance']:
            if f_bonds_prod is not None:
                f_atoms_diff = f_atoms_prod - f_atoms_reac
                f_bonds_diff = f_bonds_prod - f_bonds_reac
            else:
                f_atoms_diff = f_atoms_reac
                f_bonds_diff = f_bonds_reac
                f_atoms_prod = f_atoms_reac
                f_bonds_prod = f_bonds_reac
        
        if self.args.reaction_mode in ['reac_prod', 'reac_prod_balance']:
            f_atoms = torch.cat([f_atoms_reac, f_atoms_prod[:, :, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond_only = torch.cat([f_bonds_reac, f_bonds_prod], dim=-1)
                
        elif self.args.reaction_mode in ['reac_diff', 'reac_diff_balance']:
            f_atoms = torch.cat([f_atoms_reac, f_atoms_diff[:, :, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond_only = torch.cat([f_bonds_reac, f_bonds_diff], dim=-1)
                
        elif self.args.reaction_mode in ['prod_diff', 'prod_diff_balance']:
            f_atoms = torch.cat([f_atoms_prod, f_atoms_diff[:, :, :, PARAMS.MAX_ATOMIC_NUM+1:]], dim=-1)
            f_bond_only = torch.cat([f_bonds_prod, f_bonds_diff], dim=-1)
            
        f_bonds = torch.cat([f_atoms.unsqueeze(3).expand(-1, -1, -1, f_bond_only.shape[3], -1), f_bond_only], dim=-1)
        
        return f_atoms, f_bonds

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
        
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)
        
        f_atoms_reac, f_atoms_prod_, f_bonds_reac, f_bonds_prod_, adj_reac, adj_prod_, perm, num_atoms, a2b, b2a, b2revb, a_scope, b_scope, indicator = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms_reac, f_atoms_prod_, f_bonds_reac, f_bonds_prod_, adj_reac, adj_prod_, perm, num_atoms, a2b, b2a, b2revb, indicator = f_atoms_reac.to(self.device), f_atoms_prod_.to(self.device), f_bonds_reac.to(self.device), f_bonds_prod_.to(self.device), adj_reac.to(self.device), adj_prod_.to(self.device), perm.to(self.device), num_atoms.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device), indicator.to(self.device)
        # print("n_atoms", num_atoms)
        if self.soft_perm==True:
            if self.use_same_feat:
                self.f_atoms_reac_sep, self.f_bonds_reac_sep = self.atom_bond_feat(f_atoms_reac, f_bonds_reac)
                self.f_atoms_prod_sep, self.f_bonds_prod_sep = self.atom_bond_feat(f_atoms_prod_, f_bonds_prod_)
            else:
                self.f_atoms_reac_sep, self.f_bonds_reac_sep = f_atoms_reac, f_bonds_reac
                self.f_atoms_prod_sep, self.f_bonds_prod_sep = f_atoms_prod_, f_bonds_prod_
            atom_hiddens_r, message_bond_reac = self.rpgnn(self.f_atoms_reac_sep, self.f_bonds_reac_sep, adj_reac)
            atom_hiddens_p, message_bond_prod = self.rpgnn(self.f_atoms_prod_sep, self.f_bonds_prod_sep, adj_prod_)
            
            permutation = []
            for i in range(atom_hiddens_r.shape[1]):
                if self.args.similarity == 'dot':
                    sim = torch.einsum('bid,bdj->bij', self.FF_node(atom_hiddens_r[:, i]), self.FF_node(atom_hiddens_p[:, i]).mT)
                    
                elif self.args.similarity == 'max':
                    r_mat = self.FF_node(atom_hiddens_r[:, i]).unsqueeze(2).expand(-1, -1, atom_hiddens_r.shape[2], -1)
                    p_mat = self.FF_node(atom_hiddens_p[:, i]).unsqueeze(1).expand(-1, atom_hiddens_r.shape[2], -1, -1)
                    sim = - torch.sum(torch.maximum(r_mat, p_mat), dim=-1)
                    # sim = - torch.maximum(torch.sum(r_mat, dim=-1), torch.sum(p_mat, dim=-1))
                permutation.append(self.pytorch_sinkhorn_iters(sim, val_perm, indicator, ptype='node').unsqueeze(1))
            permutation = torch.cat(permutation, dim=1)      
            if self.args.use_lrl_network:
                f_message_r = self.ntwk(message_bond_reac)
                f_message_p = self.ntwk(message_bond_prod)
                adj_reac_f = f_message_r * adj_reac.unsqueeze(-1)
                adj_prod_f_ = f_message_p * adj_prod_.unsqueeze(-1)
            else:
                adj_reac_f = self.f_bonds_reac_sep * adj_reac.squeeze(1).unsqueeze(-1)
                adj_prod_f_ = self.f_bonds_prod_sep * adj_prod_.squeeze(1).unsqueeze(-1)
            
            adj_reac = message_bond_reac * adj_reac.unsqueeze(-1)
            adj_prod_ = message_bond_prod * adj_prod_.unsqueeze(-1)

        elif self.soft_perm==False:
            permutation = perm
        elif self.soft_perm==None:
            permutation = torch.eye(perm.shape[-1]).to(self.device)
            permutation = permutation.unsqueeze(0).unsqueeze(1). expand(perm.shape[0], perm.shape[1], -1, -1)
            #permutation = permutation.unsqueeze(1).expand(-1, perm.shape[1], -1, -1)
            
        if self.args.perm_type == 'nopm':
            if self.use_same_feat:
                self.f_atoms_reac_sep, self.f_bonds_reac_sep = self.atom_bond_feat(f_atoms_reac, f_bonds_reac)
                self.f_atoms_prod_sep, self.f_bonds_prod_sep = self.atom_bond_feat(f_atoms_prod_, f_bonds_prod_)
            else:
                self.f_atoms_reac_sep, self.f_bonds_reac_sep = f_atoms_reac, f_bonds_reac
                self.f_atoms_prod_sep, self.f_bonds_prod_sep = f_atoms_prod_, f_bonds_prod_
            atom_hidden_r, message_bond_reac = self.rpgnn(self.f_atoms_reac_sep, self.f_bonds_reac_sep, adj_reac)
            atom_hidden_p, message_bond_prod = self.rpgnn(self.f_atoms_prod_sep, self.f_bonds_prod_sep, adj_prod_)
            
            atom_hiddens = self.compactNN(torch.cat([atom_hidden_r, atom_hidden_p], dim=-1))
               
        else:
            f_atoms_prod = torch.einsum('bsij,bsjd->bsid', permutation, f_atoms_prod_)  
            f_bonds_prod = torch.einsum('bsikd,bskl->bsild', torch.einsum('bsij,bsjkd->bsikd', permutation, f_bonds_prod_), permutation.mT)  
            if self.soft_perm==True:
                adj_prod_f = torch.einsum('bsikd,bskl->bsild', torch.einsum('bsij,bsjkd->bsikd', permutation, adj_prod_f_), permutation.mT)
                adj_prod = torch.einsum('bsikd,bskl->bsild', torch.einsum('bsij,bsjkd->bsikd', permutation, adj_prod_), permutation.mT)
            else:
                adj_prod = torch.einsum('bsik,bskl->bsil', torch.einsum('bsij,bsjk->bsik', permutation, adj_prod_), permutation.mT)
            
            self.f_atoms, self.f_bonds = self.atom_bond_feat(f_atoms_reac, f_bonds_reac, f_atoms_prod, f_bonds_prod)
            
            adj_u = torch.maximum(adj_reac, adj_prod)
            
            if self.soft_perm==True:
                adj_u_f =  torch.sigmoid(torch.maximum(adj_reac_f, adj_prod_f))
                if self.use_sep_gnn:
                    atom_hiddens, message_bond = self.tgnn(self.f_atoms, self.f_bonds, adj_u, adj_u_f)
                else:
                    atom_hiddens, message_bond = self.rpgnn(self.f_atoms, self.f_bonds, adj_u, adj_u_f)
            else:
                if self.use_sep_gnn:
                    atom_hiddens, message_bond = self.tgnn(self.f_atoms, self.f_bonds, adj_u)
                else:
                    atom_hiddens, message_bond = self.rpgnn(self.f_atoms, self.f_bonds, adj_u)
                
            
        if self.use_node_tf:
            last_size = atom_hiddens.size()
            ohe_pos = torch.zeros(last_size[0],last_size[1], last_size[2],last_size[1]).to(self.device)
            for p in range(self.num_of_steps): 
                ohe_pos[:, p, :, p] = 1.0
            atom_hiddens = self.reducer(atom_hiddens)
            atom_hiddens = torch.cat([atom_hiddens, ohe_pos], dim=-1)
            atom_hiddens = atom_hiddens.view(last_size[0], -1, last_size[-1])
            atom_hiddens = self.encoder_layer(atom_hiddens)
            #atom_hiddens = self.transformer_encoder(atom_hiddens)
            atom_hiddens = atom_hiddens.view(last_size[0], last_size[1], -1, last_size[-1])
        
        if self.aggregation == 'set2set':
            mol_vecs = []
            for l in range(atom_hiddens.shape[1]):
                mol_vecs.append(self.node_readout(atom_hiddens[:,l]).unsqueeze(1))
            mol_vecs = torch.cat(mol_vecs, 1)
            mol_vecs = self.node_condenser(mol_vecs)
            
        
        elif self.aggregation == 'mean':
            mol_vecs_ = atom_hiddens.sum(dim=2)
            num_atoms_ = num_atoms.unsqueeze(2)
            mol_vecs = mol_vecs_ / num_atoms_
        
        elif self.aggregation == 'sum':
            mol_vecs = mol_vec.sum(dim=1)
            
        elif self.aggregation == 'norm':
            mol_vecs = mol_vec.sum(dim=2).squeeze(1) / self.aggregation_norm
            
        if self.use_step_tf:
            last_size = mol_vecs.size()
            if self.step_tf == 'Transformer':
                ohe_pos = torch.zeros(last_size[0],last_size[1], last_size[1]).to(self.device)
                for p in range(self.num_of_steps): 
                    ohe_pos[:, p, p] = 1.0
                mol_vecs = self.reducer(mol_vecs)
                mol_vecs = torch.cat([mol_vecs, ohe_pos], dim=-1)
            #h0 = torch.ones(self.num_gru_layers, mol_vecs.size()[0],mol_vecs.size()[2]).to(self.device)
            #mol_vecs, _ = self.encoder_layer(mol_vecs, h0)
            mol_vecs = self.encoder_layer(mol_vecs)
            #atom_hiddens = self.transformer_encoder(atom_hiddens)
            #mol_vecs = torch.cat([mol_vecs[:, k] for k in range(mol_vecs.shape[1])], -1)
            #mol_vecs = self.final(mol_vecs.permute(0, 2, 1)).squeeze(-1)
            
        if self.step_aggregation == 'set2set':
            mol_vecs_readout = self.graph_readout(mol_vecs)
            mol_vecs = self.graph_condenser(mol_vecs_readout)
            
        elif self.step_aggregation == 'capsule':
        
            mol_vecs_ = squash(mol_vecs, dim=2)
            mol_vecs = self.digits(mol_vecs_).reshape(-1,self.hidden_size)
            
        else:
            mol_vecs = mol_vecs.sum(1)
            
        return mol_vecs, self.regularizer(permutation, indicator)  # num_molecules x hidden


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
