import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import CanonicalAtomFeaturizer
from .chemprop_feat import ChempropFeaturizer
from .utils import load_dataset, collate_molgraphs_my, EarlyStopping, MoleculeSampler, arg_parse, Rank, run_a_train_epoch_my, run_an_eval_epoch_my, load_data, collate_molgraphs_new
from .model_new import DeepReac
import argparse

def pred_deepreac(args):

    #args = arg_parse()
    if args.gpu == "cpu":
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    train_file = args.train_file.replace('\r', '')
    test_file = args.test_file.replace('\r', '')
    pred_file = args.pred_file.replace('\r', '')
    
    train_set, c_num, mean, std = load_data(train_file, y_standard=1, datatype='train')
    test_set, c_num, _, _  = load_data(test_file, y_standard=1)
    
    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=args.seed)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                              collate_fn=collate_molgraphs_new)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_molgraphs_new)
                             
    loss_fn = nn.MSELoss(reduction='none')
    in_feats_dim = ChempropFeaturizer().feat_size('h')

    torch.manual_seed(args.pytorch_seed)
    
    model = DeepReac(in_feats_dim, len(train_set[0][1]), c_num, hidden_feats_0=[args.hidden_size]*args.n_layers, hidden_feats_1=[args.hidden_size]*args.n_layers, 
            num_heads_0=[args.num_heads]*args.n_layers, num_heads_1=[args.num_heads]*args.n_layers, out_dim=args.hidden_size, device=device)
    print(model)
    model.to(device)     
    model_path = args.ckpt_dir.replace('\r', '')
    # torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))
    test_score, out_feat_un, index_un, label_un, predict_un = run_an_eval_epoch_my(model, test_loader, mean, std, args, device)
    predict_arr = predict_un.data.cpu().numpy()
    test_df = pd.read_csv(test_file)
    test_df['predict'] = predict_arr
    test_df.to_csv(pred_file, index=False)
    # label_ratio = len(labeled)/len(data)

    # print("Size of labelled dataset:",100*label_ratio,"%")
    print("Model performance on test dataset: RMSE:", test_score[0], ";MAE:", test_score[1], ";R^2:", test_score[2])
    train_set_name = train_file.rsplit('/', 1)[1]
    test_set_name = test_file.rsplit('/', 1)[1]
    return predict_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, default=None, help='Specify the train file, such as Buchward directory')
    parser.add_argument('--test_file', type=str, default=None, help='Specify the test file')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Specify the model path')
    parser.add_argument('--pred_file', type=str, default=None, help='Specify the predict file')
    parser.add_argument("--batch_size", type=int, default=8, help="batch size to train.")
    parser.add_argument('--hidden_size', type=int, default=20, help='Hidden Size')
    parser.add_argument('--n_layers', type=float, default=2, help='number of layers')
    parser.add_argument('--num_heads', type=float, default=1, help='number of heads')
    parser.add_argument("--gpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=0, help="seed.")
    parser.add_argument("--pytorch_seed", type=int, default=0, help="pytorch_seed.")
    
    pred_deepreac(parser.parse_args())
