import os
import sys
import logging
import copy
import time

import numpy as np
import torch
from utils import factory
from utils.My_dataset import MyDataSet
from utils.toolkit import count_parameters
from utils.clip_classifier import clip_classifier,build_cache_model
from utils.class_names import datasets_info
from torch.utils.data import DataLoader
from models.clip.prompt_learner import load_clip_to_cpu, cfgc, cfgc_vitb32

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):
    logfilename = './logs/{}_{}_{}_'.format(args['prefix'], args['net_type'],args['model_name']) + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    _set_random()
    _set_device(args)
    print_args(args)


    # load clip model
    if args["backbone"] == "vitb16":
        cfg = cfgc()
    elif args["backbone"] == "vitb32":
        cfg = cfgc_vitb32()
    clip_model = load_clip_to_cpu(cfg)
    args["clip_model"] = clip_model.to(args['device'][0])
    print("\nGetting textual features as CLIP's classifier.")

    current_dataset = datasets_info[args['dataset']]
    clip_weights = clip_classifier(current_dataset['classnames'], current_dataset['template'], clip_model,
                                   args['dataset'])
    clip_weights = clip_weights.float()
    args["clip_weights"] = clip_weights

    cache_dir = os.path.join('./caches', args['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    args["cache_dir"] = cache_dir

    # Construct the cache model by few-shot training set
    select_data = np.load(args['new_dir'], allow_pickle=True).item()
    train_dataset = MyDataSet(select_data)

    cache_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,
                                   num_workers=args["num_workers"])

    print("\nConstructing cache model by few-shot test features and labels.")
    cache_keys, cache_values = build_cache_model(args, clip_model,cache_loader)
    cache_keys = cache_keys.float()
    args["cache_keys"] = cache_keys
    cache_values = cache_values.float()
    args["cache_values"] = cache_values


    model = factory.get_model(args['model_name'], args)
    model.train_phase(train_dataset,clip_weights,cache_keys,cache_values)
    logging.info('All params: {}'.format(count_parameters(model._network)))
    logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
    ckp_name = logfilename + '.pkl'
    # torch.save(model, ckp_name)

def _set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus
    print(gpus)

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))