import os
import numpy as np
import pandas as pd
from matplotlib import pyplot, patches
from argparse import ArgumentParser as ap
from collections import OrderedDict
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pkg_resources
from rxnfp.models import SmilesClassificationModel
wandb_available = False

# try:
#     import wandb
#     wandb_available = True
# except ImportError:
#     pass

def load_model(args):
    models_folder = './outputs'
    model_path = [os.path.join(models_folder, o) for o in os.listdir(models_folder) 
                        if os.path.isdir(os.path.join(models_folder,o)) and o.endswith(f'best_model')][0]
    model = SmilesClassificationModel(
        'bert', model_path, num_labels=1, args={'regression': True}, use_cuda=torch.cuda.is_available())
    return model

def make_plot(y_test, y_pred, rsme, r2_score, mae, name):
    fontsize = 16
    fig, ax = pyplot.subplots(figsize=(8,8))
    r2_patch = patches.Patch(label="R2 = {:.3f}".format(r2_score), color="#5402A3")
    rmse_patch = patches.Patch(label="RMSE = {:.1f}".format(rmse), color="#5402A3")
    mae_patch = patches.Patch(label="MAE = {:.1f}".format(mae), color="#5402A3")
    pyplot.xlim(-5,105)
    pyplot.ylim(-5,105)
    pyplot.scatter(y_pred, y_test, alpha=0.2, color="#5402A3")
    pyplot.plot(np.arange(100), np.arange(100), ls="--", c=".3")
    pyplot.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_ylabel('Measured', fontsize=fontsize)
    ax.set_xlabel('Predicted', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    return fig

def metric_wrapper(mean, std, func):
    return lambda pred, target: func(pred * std + mean, target)

def launch_training(args):
    project = None
    model_args = {
        'wandb_project': project,
        'num_train_epochs': args.num_epochs,
        'overwrite_output_dir': False,
        'learning_rate': args.lr,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'regression': True,
        'num_labels': 1,
        'fp16': False,
        'evaluate_during_training': True,
        'manual_seed': args.seed,
        'max_seq_length': 300,
        'train_batch_size': args.batch_size,
        'warmup_ratio': 0.00,
        'config': { 'hidden_dropout_prob': args.dropout }
    }
    
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    ckpt = args.ckpt
    
    if wandb_available:
        wandb.init(name=train_path, project=project, reinit=True)
    
    train_df = pd.read_csv(train_path)
    train_df.columns = ['text', 'labels']
    train_df.text = train_df.text.apply(lambda x: '.'.join(x.split('*')[:-1]) + '>>' + x.split('*')[-1])
    # train_df.labels /= 100
    val_df = pd.read_csv(val_path)
    val_df.columns = ['text', 'labels']
    val_df.text = val_df.text.apply(lambda x: '.'.join(x.split('*')[:-1]) + '>>' + x.split('*')[-1])
    # val_df.labels /= 100
    
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std
    val_df['labels'] = (val_df['labels'] - mean) / std
    
    model_path =  pkg_resources.resource_filename(
        "rxnfp", f"models/transformers/bert_{args.base_model}")
    
    pretrained_bert = SmilesClassificationModel(
        "bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
    
    # print(dir(pretrained_bert))
    '''
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__',
    '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__',
    '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
    '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
    '_create_training_progress_scores', '_get_inputs_dict', '_get_last_metrics',
    '_load_model_args', '_move_model_to_device', '_save_model', '_save_model_args',
    '_threshold', 'args', 'compute_metrics', 'config', 'device', 'eval_model', 'evaluate',
    'load_and_cache_examples', 'model', 'num_labels', 'predict', 'results', 'tokenizer',
    'train', 'train_model', 'weight']
    '''
    pretrained_bert.model.config.hidden_size = 20
    pretrained_bert.model.config.intermediate_size = 40
    
    pretrained_bert.model = pretrained_bert.model.from_pretrained(
        None, config=pretrained_bert.model.config, state_dict=OrderedDict())
    
    pretrained_bert.train_model(
        train_df, output_dir=ckpt, eval_df=val_df, mae=mean_absolute_error)
    
    if wandb_available:
        wandb.join() # multiple runs in same script
    
    test_df = pd.read_csv(test_path)
    test_df.columns = ['text', 'labels']
    test_df.text = test_df.text.apply(lambda x: '.'.join(x.split('*')[:-1]) + '>>' + x.split('*')[-1])
    # test_df.labels /= 100
    y_test = test_df.labels.values
    
    model = load_model(args)
    y_preds = model.predict(test_df.text.values)[0]
    y_preds = y_preds * std + mean
    
    r_squared = r2_score(y_test, y_preds)
    rmse = mean_squared_error(y_test, y_preds) ** 0.5
    mae = mean_absolute_error(y_test, y_preds)
    
    with open(f'{ckpt}/mae.txt', 'w') as f:
        f.write(f'{mae}')
    
    print(f"R2 {r_squared:.2f} | RMSE {rmse:.1f} | MAE {mae:.1f}")
    # fig = make_plot(y_test, y_preds, rmse, r_squared, mae, name)

if __name__ == '__main__':
    parser = ap()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--base-model', type=str)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--val-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--ckpt', type=str)

    args = parser.parse_args()
    launch_training(args)
