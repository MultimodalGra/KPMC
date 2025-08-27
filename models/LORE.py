import torch
import torch.nn as nn

from models.clip.prompt_learner import cfgc, cfgc_vitb32, load_clip_to_cpu, TextEncoder, PromptLearner_v2, PromptLearner_v4, PromptLearner_v3, PromptLearner_flower,  PromptLearner_nwpu,  PromptLearner_dog, PromptLearner_ucf, PromptLearner_ucfwithCoop, PromptLearner_v2withCoop
from utils.class_names import cifar10_classnames, cifar100_classnames, stanfordcars_classnames,  dtd_classnames, SAT_classnames, Aircraft_classnames, flower_classnames, nwpu_classnames, pattern_classnames,  imagenet_classnames, dog_classnames, ucf_classnames,caltech101_classnames
from models.WB_module import White_box_module
from utils.InfoNCE import InfoNCE
import torch.nn.functional as F
from utils.distance import distance

import copy

class LORE(nn.Module):

    def __init__(self, args):
        super(LORE, self).__init__()
        if args["backbone"] == "vitb16":
            self.cfg = cfgc()
        elif args["backbone"] == "vitb32":
            self.cfg = cfgc_vitb32()
        clip_model = args["clip_model"]
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.class_num = 1
        self.top_k = args["top_k"]
        self.args = args
        self.layer_num = self.image_encoder.layers
        if args['dataset'] == 'cifar':
            self.class_num = 100
            self.prompt_learner = PromptLearner_v2withCoop(self.cfg, cifar100_classnames, self.clip_model)
        if args['dataset'] == 'cifar10':
            self.class_num = 10
            self.prompt_learner = PromptLearner_v2withCoop(self.cfg, cifar10_classnames, self.clip_model)
        if args['dataset'] == 'cars':
            self.class_num = len(stanfordcars_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, stanfordcars_classnames, self.clip_model)
        if args['dataset'] == 'dtd':
            self.class_num = len(dtd_classnames)
            self.prompt_learner = PromptLearner_v2withCoop(self.cfg, dtd_classnames, self.clip_model)
        if args['dataset'] == 'sat':
            self.class_num = len(SAT_classnames)
            self.prompt_learner = PromptLearner_v3(self.cfg, SAT_classnames, self.clip_model)
        if args['dataset'] == 'aircraft':
            self.class_num = len(Aircraft_classnames)
            self.prompt_learner = PromptLearner_v4(self.cfg, Aircraft_classnames, self.clip_model)
        if args['dataset'] == 'flower':
            self.class_num = len(flower_classnames)
            self.prompt_learner = PromptLearner_flower(self.cfg, flower_classnames, self.clip_model)
        if args['dataset'] == 'nwpu':
            self.class_num = len(nwpu_classnames)
            self.prompt_learner = PromptLearner_nwpu(self.cfg, nwpu_classnames, self.clip_model)
        if args['dataset'] == 'pattern':
            self.class_num = len(pattern_classnames)
            self.prompt_learner = PromptLearner_nwpu(self.cfg, pattern_classnames, self.clip_model)
        if args['dataset'] == 'Imagenet':
            self.class_num = len(imagenet_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, imagenet_classnames, self.clip_model)
        if args['dataset'] == 'dog':
            self.class_num = len(dog_classnames)
            self.prompt_learner = PromptLearner_dog(self.cfg, dog_classnames, self.clip_model)
        if args['dataset'] == 'ucf':
            self.class_num = len(ucf_classnames)
            self.prompt_learner = PromptLearner_ucfwithCoop(self.cfg, ucf_classnames, self.clip_model)
        if args['dataset'] == 'caltech101':
            self.class_num = len(caltech101_classnames)
            self.prompt_learner = PromptLearner_v2(self.cfg, caltech101_classnames, self.clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.global_p = nn.Parameter(torch.randn(self.layer_num, self.args["prompt_length"], self.args["embd_dim"])) # ?
        nn.init.normal_(self.global_p, std=0.02)

        self.classifier = nn.Linear(self.image_encoder.output_dim, self.class_num, bias=True)
        self.numtask = 0

        self.relu = nn.ReLU(inplace=True)
        self.bn1_ = nn.BatchNorm1d(self.text_encoder.text_projection.shape[1])
        self.bn2_ = nn.BatchNorm1d(self.text_encoder.text_projection.shape[1])
        self.linear_projection = copy.deepcopy(self.image_encoder.conv1)
        self.WB = White_box_module(use_stochastic=False, gcn_len=self.args["prompt_length_c"],class_num=self.class_num)

        self.clip_weights = args["clip_weights"]
        self.cache_keys = args["cache_keys"]
        self.cache_values = args["cache_values"]
        self.image_adapter = ImageAdapter(self.cache_keys, self.args['dw']).cuda()
        self.text_adapter = TextAdapter(self.clip_weights, self.args['ta']).cuda()

    def forward(
            self,
            image,
            target=None,
            p_target=None
    ):
        num_shots = self.args['shot']
        num_cls = self.args['clip_weights'].shape[1]
        mask = torch.ones([num_shots * num_cls, num_shots * num_cls])

        _mask = torch.zeros([num_shots * num_cls, num_shots * num_cls])

        for i in range(num_cls):
            for j in range(num_shots):
                for k in range(num_shots):
                    r = i * num_shots + j
                    c = i * num_shots + k
                    if r == c:
                        pass
                    else:
                        mask[r, c] = 0

        for i in range(num_cls):
            for j in range(num_shots):
                for k in range(num_shots):
                    r = i * num_shots + j
                    c = i * num_shots + k
                    _mask[r, c] = 1

        mask = mask.cuda()
        _mask = _mask.cuda()

        nce_loss_1 = InfoNCE(temperature=self.args['t1'])
        nce_loss_incls = InfoNCE(temperature=self.args['t2'])
        nce_loss_2 = InfoNCE(temperature=self.args['t3'])
        nce_loss_list = [nce_loss_1,nce_loss_incls,nce_loss_2]

        logits = []


        # Generate text prompts and encode them to obtain textual features
        prompts = self.prompt_learner().to(self.global_p.device)  # Shape: 100 77 512
        prompts = torch.clamp(prompts, min=-1e8, max=1e8)
        if torch.isnan(prompts).any():
            print("[!] Warning: prompts has NaN ")

        tokenized_prompts = self.tokenized_prompts.to(self.global_p.device)  # Shape: 100, 77
        text_features = self.text_encoder(prompts, tokenized_prompts)  # Shape: class_num feature
        class_pool_key = text_features
        class_pool_key_norm = class_pool_key / class_pool_key.norm(dim=-1, keepdim=True)

        image_tokens = self.linear_projection(image.type(self.dtype))
        image_tokens = image_tokens.to(dtype=torch.float32)

        hidden_token, local_aware_p, att_tokens = self.WB(image_tokens, class_pool_key_norm.to(dtype=torch.float32))
        local_aware_p = local_aware_p.reshape(-1, self.layer_num, self.args["prompt_length_c"], self.args["embd_dim"]).type(self.dtype)
        local_aware_p = local_aware_p.permute(1, 0, 2, 3)

        image_features = self.image_encoder(image.type(self.dtype), self.global_p, local_aware_p, self.image_encoder.class_embedding)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()

        cache_logits, image_centers, _ = self.image_adapter(
            image_features,
            beta=self.args['beta'],
            cache_values=self.cache_values,
            pow_weight=self.args['iw'],
            distance_method=self.args['distance']
        )

        logit_scale = self.logit_scale.exp()
        logits.append(self.classifier(logit_scale * image_features))


        # Normalize attention tokens and hidden tokens for contrastive loss calculation
        att_tokens_norm = att_tokens / att_tokens.norm(dim=-1, keepdim=True)
        hidden_token_norm = hidden_token / hidden_token.norm(dim=-1, keepdim=True)
        n = att_tokens_norm.shape[0]  # Batch size

        texts = self.text_adapter(self.clip_weights)  # 对文字进行优化

        features = self.image_encoder(image.type(self.dtype))[:,0,:]
        features /= features.norm(dim=-1, keepdim=True)
        clip_logits = 100. * distance(image_features.float(),texts,self.args['distance'])
        clip_logits = clip_logits.float()
        cache_logits = cache_logits.float()

        cluster_logits =  logits[0] + (cache_logits * self.args['init_alpha']  +  clip_logits  * self.args['alpha2']) * self.args['c_weight']

        nce_loss_1 = nce_loss_list[0]
        nce_loss_incls = nce_loss_list[1]
        nce_loss_2 = nce_loss_list[2]

        nce_losses_1 = nce_loss_1(
            image_centers.detach(), image_centers, mask=mask) * self.args['ls1']

        margin = 0.0
        nce_losses_in = nce_loss_incls(image_centers.detach(),
                                       image_centers,
                                       mask=_mask,
                                       margin=margin) * self.args['ls2']

        nce_losses_2 = nce_loss_2(texts.t().detach(),
                                  texts.t()) * self.args['ls3']


        loss = F.cross_entropy(
            cluster_logits,
            target)  +  nce_losses_1 +  nce_losses_in +  nce_losses_2

        if target is not None:
            real_key_norm = att_tokens_norm[:, target, :].squeeze()
            target_index = torch.arange(n).to(self.global_p.device)
            target_index = target_index * (n + 1)
            real_key_norm = real_key_norm.reshape(n*n, -1)[target_index]
            s = real_key_norm * hidden_token_norm  # B C
            increase_sim = torch.sum(s) / (real_key_norm.shape[0])
        else:
            increase_sim = logit_scale

        if p_target is not None:
            ind = torch.where(target != p_target)
            p_key_norm = att_tokens_norm[:, p_target, :].squeeze()
            p_target_index = torch.arange(n).to(self.global_p.device)
            p_target_index = p_target_index * (n + 1)
            p_key_norm = p_key_norm.reshape(n * n, -1)[p_target_index]
            p_key = p_key_norm[ind]
            s = p_key * hidden_token_norm[ind]
            reduce_sim = torch.sum(s) / (image_features.shape[0])
        else:
            reduce_sim = logit_scale

        return {
            'cluster_logits':cluster_logits,
            'features': image_features,
            'increase_sim': increase_sim,
            'reduce_sim': reduce_sim,
            'loss':loss
        }
    def inference(self, image, target=None, p_target=None):
        logits = []
        image_tokens = self.linear_projection(image.type(self.dtype))
        image_tokens = image_tokens.to(dtype=torch.float32)
        _, local_aware_p = self.WB.inference(image_tokens)
        local_aware_p = local_aware_p.reshape(-1, self.layer_num, self.args["prompt_length_c"], self.args["embd_dim"]).type(self.dtype)
        local_aware_p = local_aware_p.permute(1, 0, 2, 3)
        image_features = self.image_encoder(image.type(self.dtype), self.global_p, local_aware_p, self.image_encoder.class_embedding)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(self.classifier(logit_scale * image_features.float()))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def get_last_layer(self,image,target):
        logits = []
        image_tokens = self.linear_projection(image.type(self.dtype))
        image_tokens = image_tokens.to(dtype=torch.float32)
        _, local_aware_p = self.WB.inference(image_tokens)
        local_aware_p = local_aware_p.reshape(-1, self.layer_num, self.args["prompt_length_c"],
                                              self.args["embd_dim"]).type(self.dtype)
        local_aware_p = local_aware_p.permute(1, 0, 2, 3)
        image_features = self.image_encoder(image.type(self.dtype), self.global_p, local_aware_p,
                                                             self.image_encoder.class_embedding)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(self.classifier(logit_scale * image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
        }
class ImageAdapter(nn.Module):

    def __init__(self, cache_keys, drop_rate=0):
        super().__init__()
        self.num_cluster = 16
        self.num_cls = cache_keys.shape[1] // self.num_cluster

        self.center = nn.Parameter(cache_keys, requires_grad=False)

        self.center_bias = nn.Parameter(torch.zeros(cache_keys.shape),
                                        requires_grad=True)

        self.weight = nn.Parameter(torch.ones(cache_keys.shape[0]),
                                   requires_grad=True)

        self.num = cache_keys.shape[1]

        self.scale = nn.Parameter(torch.ones(cache_keys.shape[1]),
                                  requires_grad=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, beta=0.0, cache_values=None, pow_weight=20, distance_method=None): # x : cls_num*shot_num,512
        weight = torch.pow(F.relu(self.weight), pow_weight) # 512,
        x = x * weight # 20,512 *  512,  -> 20,512
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6) # 归一化

        center = self.center + self.center_bias # 512,cls_num * shot_num
        center = center / (center.norm(dim=0, keepdim=True) + 1e-6) # 512,class_num * shot_num

        x = x @ center
        cls_center = center @ cache_values
        cls_center = cls_center.t()

        scale = torch.pow(F.relu(self.scale), 3)
        scale = self.dropout(scale)
        x = x * scale

        beta = int(beta)
        x = (beta * x - beta)
        x = torch.clamp(x, min=-10, max=10)
        x = torch.exp(x)


        x = x @ cache_values

        return x, center.t(), cls_center


class TextAdapter(nn.Module):

    def __init__(self, clip_weights, text_alpha):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(clip_weights.shape[1:]),
                                 requires_grad=True)

        self.scale = nn.Parameter(torch.ones(clip_weights.shape[0], 1, 1),
                                  requires_grad=True)

        self.scale_2 = nn.Parameter(torch.ones(1, clip_weights.shape[2]),
                                    requires_grad=True)

        self.llm_prompt = nn.Parameter(clip_weights[-1], requires_grad=False)

        self.alpha = text_alpha

    def forward(self, x_text):
        x_text = (x_text[:-1]).mean(dim=0)
        x_text_temp = x_text * self.alpha + self.llm_prompt * (1 - self.alpha)

        x_text = x_text_temp + self.bias
        x_text = x_text / x_text.norm(dim=-1, keepdim=True)

        return x_text.transpose(1, 0)