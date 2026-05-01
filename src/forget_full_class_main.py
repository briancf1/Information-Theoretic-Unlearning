#!/bin/python3.8

import random
import os
import json
import wandb
#import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import models
import conf
from training_utils import *

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-weight_path', type=str, required=True, help='Path to model weights. If you need to train a new model use pretrain_model.py')
parser.add_argument('-dataset', type=str, required=True, nargs='?',
                    choices=['Cifar10', 'Cifar20', 'Cifar100', 'PinsFaceRecognition'],
                    help='dataset to train on')
parser.add_argument('-classes', type=int, required=True,help='number of classes')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                    help='device override; defaults to cuda when -gpu is set, else cpu')
parser.add_argument('-data_root', type=str, default=None,
                    help='dataset root override')
parser.add_argument('-results_dir', type=str, default='results',
                    help='directory to save local result JSON files')
parser.add_argument('-zsmgm_config_path', type=str, default=None,
                    help='optional JSON file with ZS-MGM overrides')
parser.add_argument('-zsmgm_learning_rate', type=float, default=None,
                    help='override ZS-MGM learning rate')
parser.add_argument('-zsmgm_epsilon', type=float, default=None,
                    help='override ZS-MGM perturbation radius')
parser.add_argument('-zsmgm_lambda_manifold', type=float, default=None,
                    help='override ZS-MGM manifold penalty weight')
parser.add_argument('-zsmgm_k_neighbors', type=int, default=None,
                    help='override ZS-MGM proxy neighbor count')
parser.add_argument('-zsmgm_pgd_steps', type=int, default=None,
                    help='override ZS-MGM PGD step count')
parser.add_argument('-zsmgm_pgd_alpha', type=float, default=None,
                    help='override ZS-MGM PGD step size')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-method', type=str, required=True, nargs='?',
                    choices=['baseline', 'retrain','finetune','blindspot','amnesiac','UNSIR', 'ssd_tuning', 'graceful_forgetting', 'lipschitz_forgetting', 'zsmgm', 'scrub', 'gkt', 'emmn'],
                    help='select unlearning method from choice set')    
parser.add_argument('-forget_class', type=str, required=True,nargs='?',help='class to forget',
                    choices=list(conf.class_dict))
parser.add_argument('-epochs', type=int, default=1, help='number of epochs of unlearning method to use')
parser.add_argument("-seed", type=int, default=0, help="seed for runs")


def parse_args(argv=None):
    return parser.parse_args(argv)


def resolve_device(args):
    requested_device = args.device
    if requested_device is None:
        requested_device = 'cuda' if args.gpu else 'cpu'

    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but is not available in this environment.')

    if requested_device == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise ValueError('MPS requested but is not available in this environment.')

    return torch.device(requested_device)


def load_zsmgm_overrides(config_path):
    if config_path is None:
        return {}

    with open(config_path, 'r', encoding='utf-8') as config_file:
        raw = json.load(config_file)

    if not isinstance(raw, dict):
        raise ValueError('zsmgm_config_path must point to a JSON object.')

    overrides = {}
    key_map = {
        'learning_rate': 'zsmgm_learning_rate',
        'zsmgm_learning_rate': 'zsmgm_learning_rate',
        'epsilon': 'zsmgm_epsilon',
        'zsmgm_epsilon': 'zsmgm_epsilon',
        'lambda_manifold': 'zsmgm_lambda_manifold',
        'zsmgm_lambda_manifold': 'zsmgm_lambda_manifold',
        'k_neighbors': 'zsmgm_k_neighbors',
        'zsmgm_k_neighbors': 'zsmgm_k_neighbors',
        'pgd_steps': 'zsmgm_pgd_steps',
        'zsmgm_pgd_steps': 'zsmgm_pgd_steps',
        'pgd_alpha': 'zsmgm_pgd_alpha',
        'zsmgm_pgd_alpha': 'zsmgm_pgd_alpha',
    }

    for raw_key, normalized_key in key_map.items():
        if raw_key in raw:
            overrides[normalized_key] = raw[raw_key]

    return overrides


def build_zsmgm_parameters(args):
    parameters = {
        'zsmgm_learning_rate': 8.63e-3,
        'zsmgm_epsilon': 3.97e-2,
        'zsmgm_lambda_manifold': 2.42e-1,
        'zsmgm_k_neighbors': 10,
        'zsmgm_pgd_steps': 20,
        'zsmgm_pgd_alpha': 1.0 / 255.0,
    }
    parameters.update(load_zsmgm_overrides(args.zsmgm_config_path))

    cli_overrides = {
        'zsmgm_learning_rate': args.zsmgm_learning_rate,
        'zsmgm_epsilon': args.zsmgm_epsilon,
        'zsmgm_lambda_manifold': args.zsmgm_lambda_manifold,
        'zsmgm_k_neighbors': args.zsmgm_k_neighbors,
        'zsmgm_pgd_steps': args.zsmgm_pgd_steps,
        'zsmgm_pgd_alpha': args.zsmgm_pgd_alpha,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            parameters[key] = value

    parameters['zsmgm_learning_rate'] = float(parameters['zsmgm_learning_rate'])
    parameters['zsmgm_epsilon'] = float(parameters['zsmgm_epsilon'])
    parameters['zsmgm_lambda_manifold'] = float(parameters['zsmgm_lambda_manifold'])
    parameters['zsmgm_k_neighbors'] = int(parameters['zsmgm_k_neighbors'])
    parameters['zsmgm_pgd_steps'] = int(parameters['zsmgm_pgd_steps'])
    parameters['zsmgm_pgd_alpha'] = float(parameters['zsmgm_pgd_alpha'])

    return parameters

if __name__=='__main__':
    args = parse_args()

    runtime_device = resolve_device(args)
    print(f'Using device: {runtime_device}')

    # # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Check that the correct things were loaded
    if args.dataset == "Cifar20":
        assert args.forget_class in conf.cifar20_classes
    elif args.dataset == "Cifar100":
        assert args.forget_class in conf.cifar100_classes

    forget_class = conf.class_dict[args.forget_class]

    batch_size = args.b

    # Change alpha here as described in the paper
    # For PinsFaceRe-cognition, we use α=50 and λ=0.1
    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 0.5
    else:
        model_size_scaler = 1


                
    # get network
    return_activations = (args.method=='gkt')
    net = getattr(models, args.net)(num_classes=args.classes, return_activations=return_activations)
    net.load_state_dict(torch.load(args.weight_path, map_location=runtime_device))
    

    # for bad teacher
    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)
    net = net.to(runtime_device)
    unlearning_teacher = unlearning_teacher.to(runtime_device)

    # For celebritiy faces
    root = args.data_root
    if root is None:
        root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    # Scale for ViT (faster training, better performance)
    img_size = 224 if args.net == "ViT" else 32
    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True, img_size=img_size
    )
    validset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=True, img_size=img_size
    )

    # Set up the dataloaders and prepare the datasets
    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    
    full_train_dl = DataLoader(deepcopy(trainset), batch_size=args.b, shuffle=True)
    
    _sample = next(iter(trainloader))[0]
    print(_sample.min(), _sample.max(), _sample.mean())

    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)

    classwise_train, classwise_test = forget_full_class_strategies.get_classwise_ds(
        trainset, args.classes
    ), forget_full_class_strategies.get_classwise_ds(validset, args.classes)

    (
        retain_train,
        retain_valid,
        forget_train,
        forget_valid,
    ) = forget_full_class_strategies.build_retain_forget_sets(
        classwise_train, classwise_test, args.classes, forget_class
    )
    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)

    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    # full_train_dl = DataLoader(
    #     ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    #     batch_size=batch_size,
    # )
    print(len(forget_train_dl))

    if args.net == 'ViT':
        damp_val = 1
        select_val = 3.5
        lipschitz_weighting = 0.8
        learning_rate = 1.5
        #Placeholder
        scrub_alpha= 10 #1
        scrub_gamma= 10 #5
    elif args.net == 'VGG16':
        damp_val = 4
        select_val = 10
        lipschitz_weighting = 0.5
        learning_rate = 0.0003
        scrub_alpha= 1
        scrub_gamma= 5

    zsmgm_parameters = build_zsmgm_parameters(args)

    parameters = {
        "dampening_constant": 1, 
        "selection_weighting": 1,
        'eps': 0.01,
        'use_quad_weight': False,    
        'n_epochs': 1,
        "ewc_lambda": 1, 
        "frequency": [0.038,0.0475, 0.1125],
        "amplitude": 0.1,
        'lipschitz_weighting': lipschitz_weighting,
        "learning_rate": learning_rate,
        "n_samples": 25,
        **zsmgm_parameters,
    }

    kwargs = {
        'model': net,
        'unlearning_teacher':unlearning_teacher, 
        'retain_train_dl': retain_train_dl,
        'retain_valid_dl': retain_valid_dl,
        'forget_train_dl': forget_train_dl,
        'forget_valid_dl': forget_valid_dl,
        'valid_dl': validloader,
        'dampening_constant': parameters["dampening_constant"],
        'selection_weighting': parameters["selection_weighting"],
        'eps': parameters["eps"],
        'use_quad_weight': parameters['use_quad_weight'],
        'n_epochs': parameters['n_epochs'],
        'forget_class': forget_class,
        'full_train_dl': full_train_dl,
        'num_classes': args.classes,
        'dataset_name': args.dataset,
        'device': str(runtime_device),
        'model_name': args.net,
        "n_samples": parameters["n_samples"],
        'learning_rate': parameters["learning_rate"],
        "ewc_lambda": parameters["ewc_lambda"],
        "amplitude": parameters["amplitude"],
        "frequency": parameters["frequency"],
        "lipschitz_weighting": parameters['lipschitz_weighting'],
        "zsmgm_learning_rate": parameters["zsmgm_learning_rate"],
        "zsmgm_epsilon": parameters["zsmgm_epsilon"],
        "zsmgm_lambda_manifold": parameters["zsmgm_lambda_manifold"],
        "zsmgm_k_neighbors": parameters["zsmgm_k_neighbors"],
        "zsmgm_pgd_steps": parameters["zsmgm_pgd_steps"],
        "zsmgm_pgd_alpha": parameters["zsmgm_pgd_alpha"],
    }
    #############
    

    ################

    wandb.init(
        project=f"PINSFINAL_LipschitzFinal_{args.net}_{args.dataset}_fullclass",
        name=f'{args.method}_{select_val}_{args.forget_class}',
        config={
            'dataset': args.dataset,
            'net': args.net,
            'method': args.method,
            'forget_class': args.forget_class,
            'seed': args.seed,
            'device': str(runtime_device),
            **zsmgm_parameters,
        },
    )
    # Time the method
    import time

    start = time.time()

    testacc, retainacc, mia, d_f = getattr(forget_full_class_strategies, args.method)(**kwargs)

    end = time.time()
    time_elapsed = end - start
    results = {
        'TestAcc': testacc,
        'RetainTestAcc': retainacc,
        'MIA': mia,
        'df': d_f,
        'MethodTime': time_elapsed,
        'dataset': args.dataset,
        'net': args.net,
        'method': args.method,
        'forget_class': args.forget_class,
        'seed': args.seed,
        'device': str(runtime_device),
    }
    if args.method == 'zsmgm':
        results.update(zsmgm_parameters)

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir,
        f"full_class_{args.dataset}_{args.net}_{args.method}_forget-{args.forget_class}_seed-{args.seed}.json",
    )
    with open(results_path, 'w', encoding='utf-8') as results_file:
        json.dump(results, results_file, indent=2)

    wandb.log(results)
    print(json.dumps(results, indent=2))
    print(f"saved results to {results_path}")
    print("done logging...")
    wandb.finish()

