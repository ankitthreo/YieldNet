import numpy as np
import sys, csv, os
import random
import torch
from torch.utils.data import DataLoader
import dgl
from dgl.data.utils import split_dataset
import pandas as pd
from chemprop.rxn.dataset import GraphDataset, MoleculeSampler
from .util import collate_reaction_graphs
from .model_impnn import *
# from model import out_file

from argparse import ArgumentParser
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

# parser = ArgumentParser()

# parser.add_argument('--test_file', type=str, required=True,
#                     help='Test Filename')
# parser.add_argument('--train_file', type=str, default=None,
#                     help='Train Filename')
# parser.add_argument('--hidden_size', type=int, default=20,
#                     help='Hidden Size')
# parser.add_argument('--ckpt_dir', type=str, default=None,
#                     help='Checkpoint Directory')
# parser.add_argument('--gpu', type=int, default=0,
#                     help='Device')
# parser.add_argument('--batch_size', type=int, default=8,
#                     help='Batch Size')
# parser.add_argument('--pred_file', type=str, default=None, 
#                     help='Specify the predict file')

def data_finder(filename, dic):

    for key, value in dic.items():
        if filename.count(value[2])!=0:
            data = key
    return data

def setup_seed(seed): 
    random.seed(seed)                        
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    dgl.seed(seed)
    dgl.random.seed(seed)
    
ss = [436, 261, 55, 281, 346, 632, 121, 329, 364, 613]
ps = [512, 901, 76, 144, 872, 466,  24, 830, 564, 517]

data_dic = {0.0: [2161, 309, 'gp0'], 
            0.1: [4175, 596, 'gp1'], 
            0.2: [2037, 291, 'gp2'],
            1.0: [2767, 395, 'bh'],
            1.1: [350, 50, 'bhs10'],
            1.2: [350, 50, 'bhs20'],
            1.3: [350, 50, 'bhs30'],
            1.4: [350, 50, 'bhs40'],
            1.5: [350, 50, 'bhs50'],
            1.6: [350, 50, 'bhs60'],
            1.7: [350, 50, 'bhs70'],
            1.8: [350, 50, 'bhs80'],
            1.9: [350, 50, 'bhs90'],
            4.0: [518, 74, 'df'],
            4.10:[140, 20, 'dfs10'],
            4.25:[140, 20, 'dfs25'],
            4.50:[140, 20, 'dfs50'],
            4.75:[140, 20, 'dfs75'],
            4.90:[140, 20, 'dfs90'],
            5.0: [719, 103, 'ns'],
            5.25:[210, 30, 'nss25'],
            5.40:[210, 30, 'nss40'],
            5.50:[210, 30, 'nss50'],
            5.60:[210, 30, 'nss60'],
            5.75:[210, 30, 'nss75'],
            6.0: [337, 48, 'sc']}
            
   
def pred_g1(args):
    
    test_file = args.test_file
    
    if args.train_file is None:
        train_file = args.test_file.replace('test', 'train')
    else:
        train_file = args.train_file
    cuda = torch.device(f'cuda:{args.gpu}')
    split = int(test_file.split('.csv')[0][-1])
    
    s = ss[split-1]
    p = ps[split-1]

    setup_seed(p)


    data_id = data_finder(test_file, data_dic) #data_id 1: Buchwald-Hartwig, #data_id 2: Suzuki-Miyaura, %data_id 3: out-of-sample test splits for Buchwald-Hartwig
    args.data = data_id
    # split_id = 0 #data_id 1 & 2: 0-9, data_id 3: 1-4 
    train_size = data_dic[data_id][0] #data_id 1: [2767, 1977, 1186, 791, 395, 197, 98], data_id 2: [4032, 2880, 1728, 1152, 576, 288, 144], data_id 3: [3057, 3055, 3058, 3055], 4: [518], 5: [719]
    batch_size = args.batch_size    #32
    val_size = data_dic[data_id][1]  #74 #103
    use_saved = True
    # model_path = './model/model_%d_%d_%d.pt' %(data_id, split_id, train_size)
    # if not os.path.exists('./model/'): os.makedirs('./model/')

    split_id = split - 1
    if args.ckpt_dir is None:
        model_path = test_file.split('dataset')[0]+'other_models/g1/'+data_dic[data_id][2]+'/h%d/min_val_model_one_hot_4EE_bs16_%d_%d_%d/' %(args.hidden_size, data_id, split_id, train_size)
    else:
        model_path = args.ckpt_dir
    if not os.path.exists(model_path): os.makedirs(model_path)      
    train_set = GraphDataset(data_id, split_id, train_file)
    test_set = GraphDataset(data_id, split_id, test_file)
    #train_frac_split = (train_size + 1e-5)/len(data)
    #val_frac_split = (val_size + 1e-5)/len(data)
    #train_set, val_set, test_set = split_dataset(data, [train_frac_split, val_frac_split, 1 - train_frac_split - val_frac_split], shuffle = False)
    # train_set, test_set = split_dataset(data, [frac_split, 1 - frac_split], shuffle=False)

    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=s)
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=False, sampler=train_sampler, collate_fn=collate_reaction_graphs, drop_last=False)
    #val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)


    print('-- CONFIGURATIONS')    
    print('--- data_type:', data_id, split_id)
    print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
    print('--- max no. reactants:', train_set.rmol_max_cnt)
    print('--- max no. intermediates:', train_set.imol_max_cnt)
    print('--- max no. products:', train_set.pmol_max_cnt)
    print('--- use_saved:', use_saved)
    print('--- model_path:', model_path)

    # training
    try: 
        train_y = train_loader.dataset.dataset.yld
    except:
        train_y = train_loader.dataset.yld
    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)

    node_dim = train_set.rmol_node_attr[0].shape[1]
    edge_dim = train_set.rmol_edge_attr[0].shape[1]
    net = reactionMPNN(node_dim, edge_dim, train_set.rmol_max_cnt, train_set.imol_max_cnt, args, readout_feats=args.hidden_size, predict_hidden_feats=args.hidden_size).to(cuda)

    if use_saved == False:
        print('-- TRAINING')
        out_file = None
        net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args)
        #torch.save(net.state_dict(), model_path)
        model_path_init = model_path + f'init_model.pt'
        torch.save(net.state_dict(), model_path_init)
    else:
        print('-- LOAD SAVED MODEL')
        model_path_best = model_path + 'best_val.pt'
        net.load_state_dict(torch.load(model_path_best))


    # inference
    test_y = test_loader.dataset.yld

    test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 5)
    test_y_pred = np.clip(test_y_pred, 0, 100)
    pred = pd.DataFrame({'predict': test_y_pred})
    pred['Output'] = test_y
    if args.pred_file is None:
        pred.to_csv(f'./prediction/{data_dic[data_id][2]}_test_{split}_h{args.hidden_size}_{args.extra}.csv')
    else:
        pred.to_csv(args.pred_file)
    result = [mean_absolute_error(test_y, test_y_pred),
            mean_squared_error(test_y, test_y_pred) ** 0.5,
            r2_score(test_y, test_y_pred),
            stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
            
    print('-- RESULT')
    print('--- test size: %d' %(len(test_y)))
    print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))
    
    return test_y_pred

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--test_file', type=str, required=True,
                        help='Test Filename')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Train Filename')
    parser.add_argument('--hidden_size', type=int, default=20,
                        help='Hidden Size')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint Directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Device')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--pred_file', type=str, default=None, 
                        help='Specify the predict file')
    
    pred_g1(parser.parse_args())
