import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.clip import *
def clip_classifier(classnames, template, clip_model, dataset='stanford_cars'):
    dataset2prompt = {
        'cars': 'CuPL_prompts_stanfordcars.json',
        'fgvc': 'CuPL_prompts_fgvcaircraft.json',
        'flower': 'CuPL_prompts_flowers102.json',
        'oxford_pets': 'CuPL_prompts_oxfordpets.json',
        'food101': 'CuPL_prompts_food101.json',
        'sun397': 'CuPL_prompts_sun397.json',
        'sat': 'CuPL_prompts_eurosat.json',
        'caltech101': 'CuPL_prompts_caltech101.json',
        'dtd': 'CuPL_prompts_dtd.json',
        'ucf': 'CuPL_prompts_ucf101.json',
        'Imagenet': 'CuPL_prompts_imagenet.json',
        'nwpu':'res45_prompts_full.json',
        'aircraft':'airplane_prompts_full.json',
        'cifar':'cifar100_prompts_full.json',
        'cifar10':'cifar10_prompts_full.json',
        'dog':'prompt_dog.json',
        'pattern':'prompt_pattern.json'
    }
    print(dataset2prompt[dataset])
    f = open('/home/hh/code/KGPT_b/gpt3_prompts/' + dataset2prompt[dataset])

    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts
            texts_token = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts_token)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            L = len(template_texts)


            total_len = 16
            embedding_len = class_embeddings.shape[0]
            embedding_len - total_len
            distance = (class_embeddings.shape[0] - L) // (total_len - L)
            for i in range(total_len - L):
                left = L + i * distance
                right = L + (i + 1) * distance
                if i == total_len - L - 1:
                    right = class_embeddings.shape[0]
                embeddings = class_embeddings[left:right, :].mean(
                    dim=0).unsqueeze(0)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                class_embeddings[L + i] = embeddings * 1.0

            clip_weights.append(class_embeddings[:total_len])

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_pre_feat'] == "False":

        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(
                    augment_idx, cfg['augment_epoch']))
                for i, batch in enumerate(tqdm(train_loader_cache)):
                    images, target = batch[0], batch[1]
                    images = images.cuda() # b,3,224,224
                    image_features = clip_model.encode_image(images) # 256,512
                    image_features = image_features[:,0,:]
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(
            cache_keys,
            cfg['cache_dir'] + '/keys_' + str(cfg['shot']) + "shots.pt")
        torch.save(
            cache_values,
            cfg['cache_dir'] + '/values_' + str(cfg['shot']) + "shots.pt")
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' +
                                str(cfg['shot']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' +
                                  str(cfg['shot']) + "shots.pt")
    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    print("load_pre_feat",cfg['load_pre_feat'])
    if cfg['load_pre_feat'] == "False":
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features[:,0,:]
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels
