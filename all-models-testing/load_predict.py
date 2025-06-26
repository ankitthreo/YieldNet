import sys
from math import sqrt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from time import time

seeds = np.genfromtxt('seeds.txt')

sys.path.insert(0, './chemprop')

def prep_chemprop(folder, split, hidden):
    from chemprop.args import TrainArgs
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    args = TrainArgs().parse_args()
    args.data_path = f'./dataset/{folder}/train/{filename}'
    args.separate_val_path = f'./dataset/{folder}/val/{filename}'
    args.separate_test_path = f'./dataset/{folder}/test/{filename}'
    if 'gpy' in folder:
        args.use_node_tf = False
    else:
        args.use_node_tf = True
    args.save_dir = f'./models_{hidden}/chemprop/{folder}'
    args.gpu = 3
    args.dataset_type = 'regression'
    args.metric = 'mae'
    args.extra_metrics = ['rmse', 'r2']
    args.reaction = True
    args.reaction_mode = 'reac_diff'
    args.explicit_h = True
    args.use_lrl_network = True
    args.perm_type = 'soft'
    args.similarity = 'max'
    args.gumbel_noise_factor = 1.0
    args.gumbel_temperature = 0.1
    args.sinkhorn_iters = 10
    args.perm_regularizer = 'squared_fro_norm_inv'
    args.perm_reg_lambda = 0.1
    args.loss_agg_func = 'sum'
    args.epochs = 100
    args.init_lr = 0.0001
    args.max_lr = 0.001
    args.final_lr = 0.0001
    args.aggregation = 'set2set'
    args.step_aggregation = 'set2set'
    args.seed = seeds[2 * (split - 1)]
    args.batch_size = batch_size
    args.ffn_hidden_size = hidden
    args.hidden_size = hidden
    args.pytorch_seed = seeds[2 * split - 1]
    args.process_args()
    model = f'./models/chemprop/{folder}/fold_0/cv{split}-h{hidden}-model.pt'
    
    return args, model

def run_chemprop(test_args, model_path):
    
    from collections import defaultdict
    import csv
    import json
    from logging import Logger
    import os
    import sys
    from typing import Callable, Dict, List, Tuple
    import subprocess

    import numpy as np
    import pandas as pd

    from chemprop.args import TrainArgs
    from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
    from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
    from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
    from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters

    @timeit(logger_name=TRAIN_LOGGER_NAME)
    def cross_validate(args: TrainArgs,
                    train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                    ) -> Tuple[float, float]:
        """
        Runs k-fold cross-validation.

        For each of k splits (folds) of the data, trains and tests a model on that split
        and aggregates the performance across folds.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                    loading data and training the Chemprop model.
        :param train_func: Function which runs training.
        :return: A tuple containing the mean and standard deviation performance across folds.
        """
        logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Initialize relevant variables
        init_seed = args.seed
        save_dir = args.save_dir
        args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                        target_columns=args.target_columns, ignore_columns=args.ignore_columns)

        # set explicit H option and reaction option
        reset_featurization_parameters(logger=logger)
        set_explicit_h(args.explicit_h)
        set_adding_hs(args.adding_h)
        set_keeping_atom_map(args.keeping_atom_map)
        if args.reaction:
            set_reaction(args.reaction, args.reaction_mode)  ###changed
        elif args.reaction_solvent:
            set_reaction(True, args.reaction_mode)  ###changed
        
        # Get data
        debug('Loading data')
        data = get_data(
            path=args.data_path,
            args=args,
            logger=logger,
            skip_none_targets=True,
            data_weights_path=args.data_weights_path
        )
        validate_dataset_type(data, dataset_type=args.dataset_type)
        args.features_size = data.features_size()

        if args.atom_descriptors == 'descriptor':
            args.atom_descriptors_size = data.atom_descriptors_size()
        elif args.atom_descriptors == 'feature':
            args.atom_features_size = data.atom_features_size()
            set_extra_atom_fdim(args.atom_features_size)
        if args.bond_descriptors == 'descriptor':
            args.bond_descriptors_size = data.bond_descriptors_size()
        elif args.bond_descriptors == 'feature':
            args.bond_features_size = data.bond_features_size()
            set_extra_bond_fdim(args.bond_features_size)
        
        args.num_of_steps = data.number_of_steps
        debug(f'Number of tasks = {args.num_tasks}')

        if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
            raise ValueError('The number of provided target weights must match the number and order of the prediction tasks')

        # Run training on different random seeds for each fold
        all_scores = defaultdict(list)
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
            data.reset_features_and_targets()

            # If resuming experiment, load results from trained models
            test_preds, test_targets, params, tt = train_func(args, data)

        return test_preds, test_targets, params, tt

    import json
    from logging import Logger
    import os
    from typing import Dict, List

    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    import pandas as pd
    from tensorboardX import SummaryWriter
    import torch
    from tqdm import trange
    from torch.optim.lr_scheduler import ExponentialLR

    from chemprop.train.evaluate import evaluate, evaluate_predictions
    from chemprop.train.predict import predict
    from chemprop.train.train import train
    from chemprop.train.loss_functions import get_loss_func
    from chemprop.spectra_utils import normalize_spectra, load_phase_mask
    from chemprop.args import TrainArgs
    from chemprop.constants import MODEL_FILE_NAME
    from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
    from chemprop.models import MoleculeModel
    from chemprop.nn_utils import param_count, param_count_all
    from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
        save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean


    def run_test(args: TrainArgs,
                    data: MoleculeDataset,
                    logger: Logger = None) -> Dict[str, List[float]]:
        """
        Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                    loading data and training the Chemprop model.
        :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
        :param logger: A logger to record output.
        :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

        """
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        # Split data
        debug(f'Splitting data with seed {args.seed}')
        if args.separate_test_path:
            test_data = get_data(path=args.separate_test_path,
                                args=args,
                                features_path=args.separate_test_features_path,
                                atom_descriptors_path=args.separate_test_atom_descriptors_path,
                                bond_descriptors_path=args.separate_test_bond_descriptors_path,
                                phase_features_path=args.separate_test_phase_features_path,
                                constraints_path=args.separate_test_constraints_path,
                                smiles_columns=args.smiles_columns,
                                loss_function=args.loss_function,
                                logger=logger)
        if args.separate_val_path:
            val_data = get_data(path=args.separate_val_path,
                                args=args,
                                features_path=args.separate_val_features_path,
                                atom_descriptors_path=args.separate_val_atom_descriptors_path,
                                bond_descriptors_path=args.separate_val_bond_descriptors_path,
                                phase_features_path=args.separate_val_phase_features_path,
                                constraints_path=args.separate_val_constraints_path,
                                smiles_columns=args.smiles_columns,
                                loss_function=args.loss_function,
                                logger=logger)

        if args.separate_val_path and args.separate_test_path:
            train_data = data
        elif args.separate_val_path:
            train_data, _, test_data = split_data(data=data,
                                                split_type=args.split_type,
                                                sizes=args.split_sizes,
                                                key_molecule_index=args.split_key_molecule,
                                                seed=args.seed,
                                                num_folds=args.num_folds,
                                                args=args,
                                                logger=logger)
        elif args.separate_test_path:
            train_data, val_data, _ = split_data(data=data,
                                                split_type=args.split_type,
                                                sizes=args.split_sizes,
                                                key_molecule_index=args.split_key_molecule,
                                                seed=args.seed,
                                                num_folds=args.num_folds,
                                                args=args,
                                                logger=logger)
        else:
            train_data, val_data, test_data = split_data(data=data,
                                                        split_type=args.split_type,
                                                        sizes=args.split_sizes,
                                                        key_molecule_index=args.split_key_molecule,
                                                        seed=args.seed,
                                                        num_folds=args.num_folds,
                                                        args=args,
                                                        logger=logger)

        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                    f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.save_smiles_splits:
            save_smiles_splits(
                data_path=args.data_path,
                save_dir=args.save_dir,
                task_names=args.task_names,
                features_path=args.features_path,
                constraints_path=args.constraints_path,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                smiles_columns=args.smiles_columns,
                logger=logger,
            )

        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
            val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        else:
            atom_descriptor_scaler = None

        if args.bond_descriptor_scaling and args.bond_descriptors is not None:
            bond_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
            val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
            test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        else:
            bond_descriptor_scaler = None

        args.train_data_size = len(train_data)

        debug(f'Total size = {len(data):,} | '
            f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

        if len(val_data) == 0:
            raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
                a validation set is required to conduct early stopping according to the selected evaluation metric. This \
                may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

        if len(test_data) == 0:
            debug('The test data split is empty. This may be either because splitting with no test set was selected, \
                such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
                Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
            empty_test_set = True
        else:
            empty_test_set = False

        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            if args.is_atom_bond_targets:
                scaler = None
                atom_bond_scaler = train_data.normalize_atom_bond_targets()
            else:
                scaler = train_data.normalize_targets()
                atom_bond_scaler = None
            args.spectra_phase_mask = None
        elif args.dataset_type == 'spectra':
            debug('Normalizing spectra and excluding spectra regions based on phase')
            args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
            for dataset in [train_data, test_data, val_data]:
                data_targets = normalize_spectra(
                    spectra=dataset.targets(),
                    phase_features=dataset.phase_features(),
                    phase_mask=args.spectra_phaseloss_fun_mask,
                    excluded_sub_value=None,
                    threshold=args.spectra_target_floor,
                )
                dataset.set_targets(data_targets)
            scaler = None
            atom_bond_scaler = None
        else:
            args.spectra_phase_mask = None
            scaler = None
            atom_bond_scaler = None

        # Set up test set evaluation
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        elif args.is_atom_bond_targets:
            sum_test_preds = []
            for tb in zip(*test_data.targets()):
                tb = np.concatenate(tb)
                sum_test_preds.append(np.zeros((tb.shape[0], 1)))
            sum_test_preds = np.array(sum_test_preds, dtype=object)
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        # Automatically determine whether to cache
        if len(data) <= args.cache_cutoff:
            set_cache_graph(True)
            num_workers = 0
        else:
            set_cache_graph(False)
            num_workers = args.num_workers

        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )
        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=num_workers
        )
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=num_workers
        )

        if args.class_balance:
            debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

        # Train ensemble of models
        for model_idx in range(args.ensemble_size):
            # Evaluate on test set using model with best validation score
            model = load_checkpoint(model_path, device=args.device)
            model.encoder.set_mode('test')
            if empty_test_set:
                info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
            else:
                start = time()
                test_preds = predict(
                    model=model,
                    data_loader=test_data_loader,
                    scaler=scaler,
                    atom_bond_scaler=atom_bond_scaler,
                )
                end = time()
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    is_atom_bond_targets=args.is_atom_bond_targets,
                    gt_targets=test_data.gt_targets(),
                    lt_targets=test_data.lt_targets(),
                    logger=logger
                )

                if len(test_preds) != 0:
                    if args.is_atom_bond_targets:
                        sum_test_preds += np.array(test_preds, dtype=object)
                    else:
                        sum_test_preds += np.array(test_preds)

                # Average test score
                for metric, scores in test_scores.items():
                    avg_test_score = np.nanmean(scores)
                    info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                    if metric == 'mae':
                        ret_val = avg_test_score

        return test_preds, test_targets, sum(param.numel() for param in model.parameters()), end - start
    
    return cross_validate(args=test_args, train_func=run_test)

def data_finder(filename, dic):
        for key, value in dic.items():
            if filename.count(value[2])!=0:
                data = key
        return float(data)
        
def prep_g1(folder, split, hidden):
    
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    
    args = parser.parse_args()
    args.train_file = f'./dataset/{folder}/train/{filename}'
    args.test_file = f'./dataset/{folder}/test/{filename}'
    args.hidden_size = hidden
    args.ckpt_dir = f'./models/g1/{folder}'
    args.batch_size = batch_size
    
    data_dic = {0.0: [2161, 309, 'gp0'], 
            0.1: [4175, 596, 'gp1'], 
            0.2: [2037, 291, 'gp2'],
            11.0: [2161, 309, 'gpy0'], 
            12.0: [4175, 596, 'gpy1'], 
            13.0: [2037, 291, 'gpy2'],
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
            6.0: [337, 48, 'sc'],
            10.0:[805, 115, 'uspto']}
    
    data_id = data_finder(args.test_file, data_dic)
    args.data = data_id
    train_size = data_dic[data_id][0]
    batch_size = args.batch_size
    val_size = data_dic[data_id][1]
    split_id = split - 1
    args.split_id = split_id
    args.train_size = train_size
    
    
    model = f'./models/g1/' + data_dic[data_id][2] + f'/min_val_model_one_hot_4EE_bs16_{int(data_id)}_{split_id}_{train_size}/'
    
    return args, model

def run_g1(test_args, model_path):
    import numpy as np
    import sys, csv, os
    import random
    import torch
    from torch.utils.data import DataLoader
    import dgl
    from dgl.data.utils import split_dataset
    import pandas as pd
    from chemprop.rxn.dataset import GraphDataset, MoleculeSampler
    from G1.util import collate_reaction_graphs
    from G1.model_impnn import reactionMPNN, inference

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy import stats
    
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
    args = test_args
    
    test_file = args.test_file
    use_saved = True
    
    if args.train_file is None:
        train_file = args.test_file.replace('test', 'train')
    else:
        train_file = args.train_file
    cuda = torch.device(f'cuda:{args.gpu}')
    split = int(test_file.split('.csv')[0][-1])
    
    s = ss[split-1]
    p = ps[split-1]

    setup_seed(p)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    train_set = GraphDataset(args.data, args.split_id, train_file)
    test_set = GraphDataset(args.data, args.split_id, test_file)

    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=s)
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([args.batch_size, len(train_set)])), shuffle=False, sampler=train_sampler, collate_fn=collate_reaction_graphs, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

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
        out_file = None
        net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args)
        #torch.save(net.state_dict(), model_path)
        model_path_init = model_path + f'init_model.pt'
        torch.save(net.state_dict(), model_path_init)
    else:
        model_path_best = model_path + 'best_val.pt'
        print(model_path_best)
        net.load_state_dict(torch.load(model_path_best))

    # inference
    test_y = test_loader.dataset.yld

    start = time()
    test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 5)
    test_y_pred = np.clip(test_y_pred, 0, 100)
    end = time()

    pred = pd.DataFrame({'predict': test_y_pred})
    pred['Output'] = test_y
    result = [mean_absolute_error(test_y, test_y_pred),
            mean_squared_error(test_y, test_y_pred) ** 0.5,
            r2_score(test_y, test_y_pred),
            stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
            
    return test_y_pred, test_y, sum(param.numel() for param in net.parameters()), end - start

def prep_g2(folder, split, hidden):
    
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    
    args = parser.parse_args()
    args.train_file = f'./dataset/{folder}/train/{filename}'
    args.test_file = f'./dataset/{folder}/test/{filename}'
    args.hidden_size = hidden
    args.ckpt_dir = f'./models/g2/{folder}'
    args.batch_size = batch_size
    
    data_dic = {0.0: [2161, 309, 'gp0'], 
            0.1: [4175, 596, 'gp1'], 
            0.2: [2037, 291, 'gp2'],
            11.0: [2161, 309, 'gpy0'], 
            12.0: [4175, 596, 'gpy1'], 
            13.0: [2037, 291, 'gpy2'],
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
            6.0: [337, 48, 'sc'],
            10.0:[805, 115, 'uspto']}
    
    data_id = data_finder(args.test_file, data_dic)
    args.data = data_id
    train_size = data_dic[data_id][0]
    batch_size = args.batch_size
    val_size = data_dic[data_id][1]
    split_id = split - 1
    args.split_id = split_id
    args.train_size = train_size
    
    model = f'./models/g2/' + data_dic[data_id][2] + f'/min_val_model_one_hot_4EE_bs16_{int(data_id)}_{split_id}_{train_size}/'
    
    return args, model

def run_g2(test_args, model_path):
    import numpy as np
    import sys, csv, os
    import random
    import torch
    from torch.utils.data import DataLoader
    import dgl
    from dgl.data.utils import split_dataset
    import pandas as pd
    from chemprop.rxn.dataset import GraphDataset, MoleculeSampler
    from G2.util import collate_reaction_graphs
    from G2.model_impnn import reactionMPNN, inference

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy import stats
    
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
    args = test_args
    
    test_file = args.test_file
    use_saved = True
    
    if args.train_file is None:
        train_file = args.test_file.replace('test', 'train')
    else:
        train_file = args.train_file
    cuda = torch.device(f'cuda:{args.gpu}')
    split = int(test_file.split('.csv')[0][-1])
    
    s = ss[split-1]
    p = ps[split-1]

    setup_seed(p)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    train_set = GraphDataset(args.data, args.split_id, train_file)
    test_set = GraphDataset(args.data, args.split_id, test_file)

    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=s)
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([args.batch_size, len(train_set)])), shuffle=False, sampler=train_sampler, collate_fn=collate_reaction_graphs, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

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
        out_file = None
        net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args)
        #torch.save(net.state_dict(), model_path)
        model_path_init = model_path + f'init_model.pt'
        torch.save(net.state_dict(), model_path_init)
    else:
        model_path_best = model_path + 'best_val.pt'
        net.load_state_dict(torch.load(model_path_best))

    # inference
    test_y = test_loader.dataset.yld

    start = time()
    test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 5)
    test_y_pred = np.clip(test_y_pred, 0, 100)
    end = time()

    pred = pd.DataFrame({'predict': test_y_pred})
    pred['Output'] = test_y
    result = [mean_absolute_error(test_y, test_y_pred),
            mean_squared_error(test_y, test_y_pred) ** 0.5,
            r2_score(test_y, test_y_pred),
            stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
            
    return test_y_pred, test_y, sum(param.numel() for param in net.parameters()), end - start

def prep_g4(folder, split, hidden):
    
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    
    args = parser.parse_args()
    args.train_file = f'./dataset/{folder}/train/{filename}'
    args.test_file = f'./dataset/{folder}/test/{filename}'
    args.hidden_size = hidden
    args.ckpt_dir = f'./models/g4/{folder}'
    args.batch_size = batch_size
    
    data_dic = {0.0: [2161, 309, 'gp0'], 
            0.1: [4175, 596, 'gp1'], 
            0.2: [2037, 291, 'gp2'],
            11.0: [2161, 309, 'gpy0'], 
            12.0: [4175, 596, 'gpy1'], 
            13.0: [2037, 291, 'gpy2'],
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
            6.0: [337, 48, 'sc'],
            10.0:[805, 115, 'uspto']}
    
    data_id = data_finder(args.test_file, data_dic)
    args.data = data_id
    train_size = data_dic[data_id][0]
    batch_size = args.batch_size
    val_size = data_dic[data_id][1]
    split_id = split - 1
    args.split_id = split_id
    args.train_size = train_size
    
    model = f'./models/g4/' + data_dic[data_id][2] + f'/min_val_model_one_hot_4EE_bs16_{int(data_id)}_{split_id}_{train_size}/'
    
    return args, model

def run_g4(test_args, model_path):
    import numpy as np
    import sys, csv, os
    import random
    import torch
    from torch.utils.data import DataLoader
    import dgl
    from dgl.data.utils import split_dataset
    import pandas as pd
    from chemprop.rxn.dataset import GraphDataset, MoleculeSampler
    from G4.util import collate_reaction_graphs
    from G4.model_impnn import reactionMPNN, inference

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy import stats
    
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
    args = test_args
    
    test_file = args.test_file
    use_saved = True
    
    if args.train_file is None:
        train_file = args.test_file.replace('test', 'train')
    else:
        train_file = args.train_file
    cuda = torch.device(f'cuda:{args.gpu}')
    split = int(test_file.split('.csv')[0][-1])
    
    s = ss[split-1]
    p = ps[split-1]

    setup_seed(p)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    train_set = GraphDataset(args.data, args.split_id, train_file)
    test_set = GraphDataset(args.data, args.split_id, test_file)

    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=s)
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([args.batch_size, len(train_set)])), shuffle=False, sampler=train_sampler, collate_fn=collate_reaction_graphs, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

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
        out_file = None
        net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args)
        #torch.save(net.state_dict(), model_path)
        model_path_init = model_path + f'init_model.pt'
        torch.save(net.state_dict(), model_path_init)
    else:
        model_path_best = model_path + 'best_val.pt'
        net.load_state_dict(torch.load(model_path_best))

    # inference
    test_y = test_loader.dataset.yld

    start = time()
    test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 5)
    test_y_pred = np.clip(test_y_pred, 0, 100)
    end = time()

    pred = pd.DataFrame({'predict': test_y_pred})
    pred['Output'] = test_y
    result = [mean_absolute_error(test_y, test_y_pred),
            mean_squared_error(test_y, test_y_pred) ** 0.5,
            r2_score(test_y, test_y_pred),
            stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]

    return test_y_pred, test_y, sum(param.numel() for param in net.parameters()), end - start

def prep_mpnn(folder, split, hidden):
    
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    
    args = parser.parse_args()
    args.train_file = f'./dataset/{folder}/train/{filename}'
    args.test_file = f'./dataset/{folder}/test/{filename}'
    args.hidden_size = hidden
    args.ckpt_dir = f'./models/mpnn/{folder}'
    args.batch_size = batch_size
    
    data_dic = {0.0: [2161, 309, 'gp0'], 
            0.1: [4175, 596, 'gp1'], 
            0.2: [2037, 291, 'gp2'],
            11.0: [2161, 309, 'gpy0'], 
            12.0: [4175, 596, 'gpy1'], 
            13.0: [2037, 291, 'gpy2'],
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
            6.0: [337, 48, 'sc'],
            10.0:[805, 115, 'uspto']}
    
    data_id = data_finder(args.test_file, data_dic)
    args.data = data_id
    train_size = data_dic[data_id][0]
    batch_size = args.batch_size
    val_size = data_dic[data_id][1]
    split_id = split - 1
    args.split_id = split_id
    args.train_size = train_size
    
    model = f'./models/mpnn/' + data_dic[data_id][2] + f'/min_val_model_one_hot_4EE_bs16_{int(data_id)}_{split_id}_{train_size}/'
    
    return args, model

def run_mpnn(test_args, model_path):
    import numpy as np
    import sys, csv, os
    import random
    import torch
    from torch.utils.data import DataLoader
    import dgl
    from dgl.data.utils import split_dataset
    import pandas as pd
    from chemprop.rxn.dataset import GraphDataset, MoleculeSampler
    from mpnn.util import collate_reaction_graphs
    from mpnn.model_impnn import reactionMPNN, inference

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy import stats
    
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
    args = test_args
    
    test_file = args.test_file
    use_saved = True
    
    if args.train_file is None:
        train_file = args.test_file.replace('test', 'train')
    else:
        train_file = args.train_file
    cuda = torch.device(f'cuda:{args.gpu}')
    split = int(test_file.split('.csv')[0][-1])
    
    s = ss[split-1]
    p = ps[split-1]

    setup_seed(p)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    train_set = GraphDataset(args.data, args.split_id, train_file)
    test_set = GraphDataset(args.data, args.split_id, test_file)

    train_sampler = MoleculeSampler(dataset=train_set, shuffle=True, seed=s)
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([args.batch_size, len(train_set)])), shuffle=False, sampler=train_sampler, collate_fn=collate_reaction_graphs, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

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
        out_file = None
        net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, args)
        #torch.save(net.state_dict(), model_path)
        model_path_init = model_path + f'init_model.pt'
        torch.save(net.state_dict(), model_path_init)
    else:
        model_path_best = model_path + 'best_val.pt'
        net.load_state_dict(torch.load(model_path_best))

    # inference
    test_y = test_loader.dataset.yld

    start = time()
    test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std, cuda, n_forward_pass = 5)
    test_y_pred = np.clip(test_y_pred, 0, 100)
    end = time()

    pred = pd.DataFrame({'predict': test_y_pred})
    pred['Output'] = test_y
    pred.to_csv('mpnn_output.csv')
    result = [mean_absolute_error(test_y, test_y_pred),
            mean_squared_error(test_y, test_y_pred) ** 0.5,
            r2_score(test_y, test_y_pred),
            stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
            
    return test_y_pred, test_y, sum(param.numel() for param in net.parameters()), end - start

def prep_deepreac(folder, split, hidden):
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df', 'dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    parser.add_argument('--n_layers', type=float, default=2, help='number of layers')
    parser.add_argument('--num_heads', type=float, default=1, help='number of heads')
    parser.add_argument("--seed", type=int, default=0, help="seed.")
    parser.add_argument("--pytorch_seed", type=int, default=0, help="pytorch_seed.")
    
    args = parser.parse_args()
    args.train_file = f'./dataset/{folder}/train/{filename}'
    args.test_file = f'./dataset/{folder}/test/{filename}'
    args.hidden_size = hidden
    args.ckpt_dir = f'./models/deepreac/{folder}'
    args.batch_size = batch_size
    
    if folder[:2] == 'gp':
        model = f'./models/deepreac/{folder}/{folder[:2]}_{split}_test_chemprop_h{hidden}_bs{batch_size}.pt'
    elif 'uspto' in folder:
        model = f'./models/deepreac/{folder}/{folder.upper()}_{split}_test_chemprop_h{hidden}_bs{batch_size}.pt'
    else:
        model = f'./models/deepreac/{folder}/{folder[:2]}i_{split}_test_chemprop_h{hidden}_bs{batch_size}.pt'
    
    return args, model

def run_deepreac(test_args, model_path):
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
    from deepreac.chemprop_feat import ChempropFeaturizer
    from deepreac.utils import load_dataset, collate_molgraphs_my, EarlyStopping, MoleculeSampler, arg_parse, Rank, run_a_train_epoch_my, run_an_eval_epoch_my, load_data, collate_molgraphs_new
    from deepreac.model_new import DeepReac
    
    if args.gpu == "cpu":
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    
    train_file = args.train_file
    test_file = args.test_file
    
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
    model.to(device)
    
    model.load_state_dict(torch.load(model_path))
    
    start = time()
    test_score, out_feat_un, index_un, label_un, predict_un = run_an_eval_epoch_my(model, test_loader, mean, std, args, device)
    predict_arr = predict_un.data.cpu().numpy()
    end = time()
    
    test_df = pd.read_csv(test_file)
    test_df['predict'] = predict_arr
    train_set_name = train_file.rsplit('/', 1)[1]
    test_set_name = test_file.rsplit('/', 1)[1]
    
    return predict_arr, label_un.data.cpu().numpy(), sum(param.numel() for param in model.parameters()), end - start

def prep_yieldbert(folder, split, hidden):
    from argparse import ArgumentParser
    
    filename = '', ''
    
    if folder == 'gp0' or folder == 'gpy0':
        batch_size = 50
        filename = f'GP0_FullCV_{split}.csv'
    elif folder == 'gp1' or folder == 'gpy1':
        batch_size = 50
        filename = f'GP1_FullCV_{split}.csv'
    elif folder == 'gp2' or folder == 'gpy2':
        batch_size = 50
        filename = f'GP2_FullCV_{split}.csv'
    elif folder == 'sc':
        batch_size = 8
        filename = f'SC_FullCV_{split}.csv'
    elif folder in ['ns']:
        batch_size = 8
        filename = f'NSi_FullCV_{split}.csv'
    elif folder in ['nss25', 'nss40', 'nss50', 'nss60', 'nss75']:
        batch_size = 8
        filename = f'NS_FullCV_{split}.csv'
    elif folder in ['df']:
        batch_size = 8
        filename = f'DFi_FullCV_{split}.csv'
    elif folder in ['dfs10', 'dfs75', 'dfs90']:
        batch_size = 8
        filename = f'DF_FullCV_{split}.csv'
    elif folder in ['uspto']:
        batch_size = 8
        filename = f'USPTO_FullCV_{split}.csv'
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Device')
    
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
    args.experiment = 'full'
    args.base_model = 'pretrained'
    args.num_epochs = 100
    args.lr = 0.0001
    args.gradient_accumulation_steps = 1
    args.seed = seeds[2 * split - 1]
    args.batch_size = batch_size
    args.dropout = 0.0
    args.train_path = f'./dataset-yb/{folder}/train/{filename}'
    args.test_path = f'./dataset-yb/{folder}/test/{filename}'
    args.ckpt = f'./models/yieldbert/{folder}'
    
    model = f'./models/yieldbert/{folder}/cv{split}-h{hidden}-model'
    
    return args, model

def run_yieldbert(test_args, model_path):
    import os
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot, patches
    from collections import OrderedDict
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    import torch
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from rxnfp.models import SmilesClassificationModel
    
    model = SmilesClassificationModel(
        'bert', model_path, num_labels=1, args={'regression': True}, use_cuda=torch.cuda.is_available())
    args = test_args
    
    train_path = args.train_path
    test_path = args.test_path
    ckpt = args.ckpt
    
    train_df = pd.read_csv(train_path)
    if 'df' in train_path:
        train_df.columns = ['text', 'temp', 'labels']
    else:
        train_df.columns = ['text', 'labels']
    if 'gpy' in train_path:
        print("Hello World!")
    elif 'df/' in train_path or 'ns/' in train_path:
        train_df.text = train_df.text.apply(lambda x: '.'.join(x.split('*')[:3]) + '>>' + x.split('*')[-1])
    elif 'uspto' in train_path:
        train_df.text = train_df.text.apply(lambda x: '.'.join(x.split('>>')[0]) + '>>' + x.split('>>')[-1])
    else:
        train_df.text = train_df.text.apply(lambda x: '.'.join(x.split('*')[:-1]) + '>>' + x.split('*')[-1])
    # train_df.labels /= 100
    
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std
    
    test_df = pd.read_csv(test_path)
    if 'df' in train_path:
        test_df.columns = ['text', 'temp', 'labels']
    else:
        test_df.columns = ['text', 'labels']
        
    if 'gpy' in train_path:
        print("Bye World!")
    elif 'df/' in train_path or 'ns/' in train_path:
        test_df.text = test_df.text.apply(lambda x: '.'.join(x.split('*')[:3]) + '>>' + x.split('*')[-1])
    elif 'uspto' in train_path:
        test_df.text = test_df.text.apply(lambda x: '.'.join(x.split('>>')[0]) + '>>' + x.split('>>')[-1])
    else:
        test_df.text = test_df.text.apply(lambda x: '.'.join(x.split('*')[:-1]) + '>>' + x.split('*')[-1])
    
    # test_df.labels /= 100
    y_test = test_df.labels.values
    
    start = time()
    y_preds = model.predict(test_df.text.values)[0] * std + mean
    end = time()
    
    return y_preds, y_test, sum(param.numel() for param in model.model.parameters()), end - start

test_functions = {
    'mpnn': (prep_mpnn, run_mpnn),
    'g1': (prep_g1, run_g1),
    'g2': (prep_g2, run_g2),
    'g4': (prep_g4, run_g4),
    'deepreac': (prep_deepreac, run_deepreac),
    'chemprop': (prep_chemprop, run_chemprop),
    'yieldbert': (prep_yieldbert, run_yieldbert)
}

error_functions = {
    'mae': lambda p, t: mean_absolute_error(p,t),
    'rmse': lambda p, t: mean_squared_error(p,t,squared=False)
}

if __name__ == '__main__':
    
    # hidden = 20
    # model = 'deepreac'
    # prep_func, run_func = test_functions[model]
    # args, model = prep_func('gpy0', 2, hidden)
    # preds, targets, params, time_taken = run_func(args, model)
    
    mode = 'ttest'
    error_mode = 'mae'
        
    if mode == 'err':
        table = open(f'./table/{error_mode}/{mode}_all_norm20_{error_mode}.txt', 'w')
        for dataset in ['gpy0', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75', 'sc', 'df']:  
            print(dataset)
            print(dataset + ',', end='', file=table, flush=True)
            for model in ['mpnn', 'g1', 'g2', 'g4', 'deepreac', 'yieldbert', 'chemprop']: #, 'chemprop'
                print(model)
                if 'gpy' in dataset:
                    if model=='mpnn':
                        hidden = 18 # 107
                    elif model == 'g1':
                        hidden = 17 # 98
                    elif model == 'g2':
                        hidden = 20 # 116
                    elif model == 'g4':
                        hidden = 20 # 118
                    elif model=='deepreac':
                        hidden = 48 # 362
                    elif model=='yieldbert':
                        hidden = 12
                    elif model=='chemprop':
                        hidden = 20 # 100
                else:
                    if model=='mpnn':
                        hidden = 38 # 107
                    elif model == 'g1':
                        hidden = 40 # 98
                    elif model == 'g2':
                        hidden = 40 # 116
                    elif model == 'g4':
                        hidden = 40 # 118
                    elif model=='deepreac':
                        hidden = 120 # 362
                    elif model=='yieldbert':
                        hidden = 28
                    elif model=='chemprop':
                        hidden = 20 # 100
                prep_func, run_func = test_functions[model]
                errors = []
                for split in range(1,11):
                    args, model = prep_func(dataset, split, hidden)
                    preds, targets, _, _ = run_func(args, model)
                    errors.append(error_functions[error_mode](preds, targets))
                print(errors)
                print(str(round(np.mean(np.array(errors)),3)) + u"\u00B1" + str(round(np.std(np.array(errors)) / sqrt(10),3)) + ',', end='', file=table, flush=True)
            print('\n', end='', file=table, flush=True)
    
    elif mode == 'ttest':
        from scipy import stats
        table = open(f'./table/{error_mode}/{mode}_gpy_norm20_{error_mode}.txt', 'w')
        for dataset in ['gpy0', 'nss25', 'nss40', 'nss50', 'nss60', 'nss75', 'sc', 'df']:  
            print(dataset)
            print(dataset + ',', end='', file=table, flush=True)
            m0 = 'chemprop'
            h0 = 20
            prep_func0, run_func0 = test_functions[m0]
            errors0 = []
            for split in range(1,11):
                args0, model0 = prep_func0(dataset, split, h0)
                preds0, targets0, _, _ = run_func0(args0, model0)
                errors0.append(error_functions[error_mode](preds0, targets0))
            for m in ['mpnn', 'g1', 'g2', 'g4', 'deepreac', 'yieldbert', 'chemprop']: #'mpnn', 'g1', 'g2', 'g4', 'deepreac', 'chemprop'
                print(m)
                if 'gpy' in dataset:
                    if m=='mpnn':
                        hidden = 18 # 107
                    elif m == 'g1':
                        hidden = 17 # 98
                    elif m == 'g2':
                        hidden = 20 # 116
                    elif m == 'g4':
                        hidden = 20 # 118
                    elif m=='deepreac':
                        hidden = 48 # 362
                    elif m=='yieldbert':
                        hidden = 12
                    elif m=='chemprop':
                        hidden = 20 # 100
                else:
                    if m=='mpnn':
                        hidden = 38 # 107
                    elif m == 'g1':
                        hidden = 40 # 98
                    elif m == 'g2':
                        hidden = 40 # 116
                    elif m == 'g4':
                        hidden = 40 # 118
                    elif m=='deepreac':
                        hidden = 120 # 362
                    elif m=='yieldbert':
                        hidden = 28
                    elif m=='chemprop':
                        hidden = 20 # 100
                prep_func, run_func = test_functions[m]
                if m==m0:
                    errors = errors0
                else:
                    errors = []
                    for split in range(1,11):
                        args, model = prep_func(dataset, split, hidden)
                        preds, targets, _, _ = run_func(args, model)
                        errors.append(error_functions[error_mode](preds, targets))
                t_stat, p_value = stats.ttest_rel(errors0, errors)
                print(str(round(np.mean(np.array(errors)),3)) + f' | {p_value:.1e}' + ',', end='', file=table, flush=True)
            print('\n', end='', file=table, flush=True)
