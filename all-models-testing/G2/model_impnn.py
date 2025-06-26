import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import dgl
from dgl.nn.pytorch import NNConv, Set2Set, TAGConv

from .util import MC_dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

#sz = 300 #nsl4, sz=20
class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, device, hidden_feats = 20,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 20):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        self.device = device
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        #self.gnn_layer = NNConv(
        #    in_feats = hidden_feats,
        #    out_feats = hidden_feats,
        #    edge_func = edge_network,
        #    aggregator_type = 'sum'
        #)

        self.tag = TAGConv(in_feats=hidden_feats, out_feats=hidden_feats, k=2) #3
        
        self.activation = nn.ReLU()
        
        #self.gru = nn.GRU(hidden_feats, hidden_feats)
        self.gru1 = nn.Linear(hidden_feats*2, hidden_feats)
        self.gru2 = nn.Linear(hidden_feats*2, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)
        
        self.weight = torch.sigmoid(nn.Parameter(torch.ones(edge_feats.shape[0]).to(self.device))) #3
        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            # print(len(node_feats),len(edge_feats), len(hidden_feats))
            #node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)  #2
            node_feats = self.activation(self.tag(g, node_feats, edge_weight=self.weight)).unsqueeze(0)  #2
            node_feats = torch.cat([node_feats, hidden_feats], dim=-1)
            node_feats, hidden_feats = self.gru1(node_feats), self.gru2(node_feats)
            # node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, n_rmol, n_imol, args,
                 readout_feats = 20,
                 predict_hidden_feats = 20, prob_dropout = 0.1):
        
        super(reactionMPNN, self).__init__()
        self.device = args.gpu
        self.mpnn = MPNN(node_in_feats, edge_in_feats, self.device, hidden_feats=args.hidden_size, readout_feats=args.hidden_size)
        self.data = args.data
        
        if int(self.data) in [0, 11, 12, 13]:
            # for GP
            self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )
        
        elif int(self.data)==1:
            #for BH
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * (n_imol+1), predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, 2)
            )
            
        elif int(self.data)==4:
            # for DF
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * (4), predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, 2)
            )
            
        elif int(self.data) in [5, 6]:  
            #for NS, SC
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * 4, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, 2)
          )
        else:
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * 3, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(predict_hidden_feats, 2)
          )
        
        
    def forward(self, rmols, imols, pmols):
    
        rxn_graph_feats = []
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        rxn_graph_feats.append(r_graph_feats)
        
        if int(self.data)==1:
            #for BH
            i1_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[0:1]]), 0)
            rxn_graph_feats.append(i1_graph_feats)
            i2_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[1:2]]), 0)
            rxn_graph_feats.append(i2_graph_feats)
            i3_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[2:]]), 0)
            rxn_graph_feats.append(i3_graph_feats)
            
        elif int(self.data) ==4:
            #for DF 
            '''
            for mol in imols:
                i_graph_feats = torch.sum(torch.stack([self.mpnn(mol)]), 0)
                rxn_graph_feats.append(i_graph_feats)
            '''
            i1_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[0:2]]), 0)
            rxn_graph_feats.append(i1_graph_feats)
            i2_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[2:]]), 0)
            rxn_graph_feats.append(i2_graph_feats)
            
        elif int(self.data)==5:
            ##for NS
            i1_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[0:2]]), 0)
            rxn_graph_feats.append(i1_graph_feats)
            i2_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[2:]]), 0)
            rxn_graph_feats.append(i2_graph_feats)
        
        elif int(self.data)==6:
            ##for SC
            i1_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[0:1]]), 0)
            rxn_graph_feats.append(i1_graph_feats)
            i2_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[1:]]), 0)
            rxn_graph_feats.append(i2_graph_feats)
        elif int(self.data)==10:
            i_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols]), 0)
            rxn_graph_feats.append(i_graph_feats)
        
        
        
        ##for notNS
        #i1_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[0:2]]), 0)
        #i2_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in imols[2:]]), 0)
        
       
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)
        rxn_graph_feats.append(p_graph_feats) 
        
        concat_feats = torch.cat(rxn_graph_feats, 1)
        #print(concat_feats.shape)
        out = self.predict(concat_feats)

        return out[:,0], out[:,1]

        
def training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args, val_monitor_epoch = 1, n_forward_pass = 5):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    
    val_mae = []
    test_result = []
    
    cuda = torch.device(f'cuda:{args.gpu}')
    print('train size: ', train_size)
    print('train size: ', train_size, file=out_file)
    
    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        imol_max_cnt = train_loader.dataset.dataset.imol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        imol_max_cnt = train_loader.dataset.imol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.MSELoss(reduction = 'none')

    n_epochs = 100
    optimizer = Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [50, 75], gamma = 0.1, verbose = False)
    min_val_mae = 9e99
    for epoch in range(n_epochs):
        
        # training
        net.train()
        start_time = time.time()
        # print('start ', start_time)
        # print('start training epoch', n_epochs)
        for batchidx, batchdata in enumerate(train_loader):
            # print('loading -- ', batchdata)
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_imol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+imol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt+imol_max_cnt:rmol_max_cnt+imol_max_cnt+pmol_max_cnt]]
            # inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]
            
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)
            
            # pred, logvar = net(inputs_rmol, inputs_pmol)
            pred, logvar = net(inputs_rmol, inputs_imol, inputs_pmol)

            loss = loss_fn(pred, labels)
            
            pred_ = pred*train_y_std + train_y_mean
            labels_ = labels*train_y_std + train_y_mean
            loss1 = loss_fn(pred_, labels_)
            loss1 = loss1.mean()
            
            loss = (1 - 0.0) * loss.mean() + 0.0 * ( loss * torch.exp(-logvar) + logvar ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item()
            train_loss1 = loss1.detach().item()
            
        if epoch in [0] + list(range(n_epochs-5, n_epochs)):
            model_path_e = model_path + f'epoch_{epoch}.pt'
            torch.save(net.state_dict(), model_path_e)
        #model_path1 = model_path + f'{epoch}.pt'
        #torch.save(net.state_dict(), model_path1)

        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60))
        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60), file=out_file)      
        lr_scheduler.step()

        # validation
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            val_y = val_loader.dataset.dataset.yld[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(net, val_loader, train_y_mean, train_y_std, cuda, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(val_y, val_y_pred),
                      mean_squared_error(val_y, val_y_pred) ** 0.5,
                      r2_score(val_y, val_y_pred)]
            
            val_mae.append(result[0])
            
            if val_mae[-1] <= min_val_mae:
                min_val_mae = val_mae[-1]
                model_path1 = model_path + f'best_val.pt'
                torch.save(net.state_dict(), model_path1)
            
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]), file=out_file)

        if test_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]
            test_y_pred, _, _ = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(test_y, test_y_pred),
                      mean_squared_error(test_y, test_y_pred) ** 0.5,
                      r2_score(test_y, test_y_pred)]

            test_result.append(result)
                      
            print('--- test at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f \n' %(epoch, len(test_y), result[0], result[1], result[2]))
            print('--- test at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f \n' %(epoch, len(test_y), result[0], result[1], result[2]), file=out_file)

    print('training terminated at epoch %d' %epoch)
    print('training terminated at epoch %d' %epoch, file=out_file)
    idx = np.argmin(np.array(val_mae))
    result = test_result[idx]
    print('--- test at MIN validation at epoch %d MAE %.3f RMSE %.3f R2 %.3f' %(idx, result[0], result[1], result[2]))
    print('--- test at MIN validation at epoch %d MAE %.3f RMSE %.3f R2 %.3f' %(idx, result[0], result[1], result[2]), file=out_file)
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 30):

    batch_size = test_loader.batch_size
    
    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        imol_max_cnt = test_loader.dataset.dataset.imol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        imol_max_cnt = test_loader.dataset.imol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt
             
    net.eval()
    MC_dropout(net)
    
    test_y_mean = []
    test_y_var = []
    
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_imol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+imol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt+imol_max_cnt:rmol_max_cnt+imol_max_cnt+pmol_max_cnt]]
            # inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]

            mean_list = []
            var_list = []
            
            for _ in range(n_forward_pass):
                mean, logvar = net(inputs_rmol, inputs_imol, inputs_pmol)
                # mean, logvar = net(inputs_rmol, inputs_pmol)
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)
    
    return test_y_pred, test_y_epistemic, test_y_aleatoric
