from torch import optim
import logging
from tqdm import tqdm
from utils.data_manager_new import DataManager
from utils.My_dataset import MyDataSet
from utils.toolkit import tensor2numpy,cls_acc
from models.LORE import LORE
from utils.get_hard_samples import *
from utils.distance import distance

class eth_dta(object):

    def __init__(self, args):
        super().__init__()
        if args["net_type"] == "LORE":
            self._network = LORE(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        self.args = args
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_2 = args["init_lr_2"]
        self.init_weight_decay = args["init_weight_decay"]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.class_num = self._network.class_num
        self._device = args['device'][0]
        self._multiple_gpus = args['device']
        self.pull_constraint = args['pull_constraint']
        self.pull_constraint_2 = args['pull_constraint_2']
        self.new_dir = args['new_dir']
        self.shot = args['shot']
        self.ds = args['dataset']

    def train_phase(self,train_dataset, clip_weights,cache_keys,cache_values):
        data_manager = DataManager(self.args)
        train_dataset_all, test_dataset = data_manager.get_dataset()
        train_dataset_all_, _ = data_manager.get_dataset()
        self.train_loader_all = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=self.num_workers)

        self._train(self.train_loader, self.test_loader, train_dataset_all, train_dataset_all_,clip_weights,cache_keys,cache_values)


    def _train(self, train_loader, test_loader, train_dataset_all, train_dataset_all_,clip_weights,cache_keys,cache_values):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "classifier" in name:
                param.requires_grad_(True)
            if "global_p" in name:
                param.requires_grad_(True)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "WB" in name:
                param.requires_grad_(True)
            if "cross_attn" in name:
                param.requires_grad_(True)
            if "image_adapter" in name:
                param.requires_grad_(True)
            if "text_adapter" in name:
                param.requires_grad_(True)
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        all_params = list(self._network.parameters())

        optimizer = optim.SGD(all_params, momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch, eta_min=self.args['lr_min'])
        self.run_epoch = self.init_epoch
        self.train_function(train_loader, test_loader, optimizer, scheduler, train_dataset_all, train_dataset_all_,clip_weights,cache_values)


    def train_function(self, train_loader, test_loader, optimizer, scheduler, train_dataset_all, train_dataset_all_,clip_weights,cache_values):
        ########################## easy stage ##################################
        beta, alpha = self.args['init_beta'], self.args['init_alpha']
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            losses = 0.
            losses1,losses2, losses3 = 0, 0,0
            correct_samples, all_samples = 0, 0
            for i, (inputs, targets, p_targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                p_targets = p_targets.to(self._device)
                self._network.image_adapter.train()
                self._network.text_adapter.train()
                outputs = self._network(inputs, target=targets, p_target=p_targets)
                loss = outputs['loss']
                cluster_logits = outputs['cluster_logits']
                loss2 = torch.mean(outputs['increase_sim'])
                loss3 = torch.mean(outputs['reduce_sim'])
                losses1 += loss.item()
                loss = loss - self.pull_constraint * loss2 + self.pull_constraint_2 * loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses2 += self.pull_constraint * loss2.item()
                losses3 += self.pull_constraint_2 * loss3.item()
                acc = cls_acc(cluster_logits, targets)
                correct_samples += acc / 100 * len(cluster_logits)
                all_samples += len(cluster_logits)

            scheduler.step()
            print('lr', scheduler.get_lr())
            train_acc = correct_samples / all_samples
            test_acc = self._compute_accuracy(self._network, test_loader,cache_values,clip_weights,beta,alpha,self.args['alpha2'])
            info = 'Epoch {}/{} => Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}, Loss3 {:.3f}, Train_accy {:.4f}, Test_accy {:.4f}'.format(
                epoch + 1, self.run_epoch, losses / len(train_loader), losses1 / len(train_loader),
                losses2 / len(train_loader), losses3 / len(train_loader), train_acc * 100, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

        ########################## hard stage ##################################
        self._network.train()
        self.run_epoch = 10
        self.init_epoch = self.run_epoch
        prog_bar = tqdm(range(self.run_epoch))
        optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr_2,weight_decay=self.init_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch,eta_min=self.args['lr_min'])

        for _, epoch in enumerate(prog_bar):
            if epoch == 0:
                if self.ds == 'cifar':
                    new_data = get_hard_sample_cifar(self._network, train_dataset_all, train_dataset_all_, self._device,self.shot)
                elif self.ds == 'cifar10':
                    new_data = get_hard_sample_cifar10(self._network, train_dataset_all, train_dataset_all_, self._device, self.shot)
                else:
                    new_data = get_hard_sample(self._network, train_dataset_all, train_dataset_all_, self._device, self.shot)
                new_train_dataset = MyDataSet(new_data)
                train_loader = DataLoader(new_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            losses = 0.
            losses1,losses2, losses3 = 0, 0, 0
            correct_samples, all_samples = 0, 0
            for i, (inputs, targets, p_targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                p_targets = p_targets.to(self._device)
                self._network.image_adapter.train()
                self._network.text_adapter.train()
                outputs = self._network(inputs, target=targets, p_target=p_targets)
                loss = outputs['loss']
                cluster_logits = outputs['cluster_logits']
                loss2 = torch.mean(outputs['increase_sim'])
                loss3 = torch.mean(outputs['reduce_sim'])
                losses1 += loss.item()
                ##################################################################
                loss = loss - self.pull_constraint * loss2 + self.pull_constraint_2 * loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses2 += self.pull_constraint * loss2.item()
                losses3 += self.pull_constraint_2 * loss3.item()
                acc = cls_acc(cluster_logits, targets)
                correct_samples += acc / 100 * len(cluster_logits)
                all_samples += len(cluster_logits)

            scheduler.step()
            print('lr', scheduler.get_lr())
            train_acc = correct_samples / all_samples
            test_acc= self._compute_accuracy(self._network, test_loader,cache_values,clip_weights,beta,alpha,self.args['alpha2'])
            info = 'Epoch {}/{} => Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}, Loss3 {:.3f}, Train_accy {:.4f}, Test_accy {:.4f}'.format(
                epoch + 1, self.run_epoch, losses / len(train_loader), losses1 / len(train_loader),
                losses2 / len(train_loader), losses3 / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_accuracy(self, model, loader,cache_values,clip_weights,beta,alpha1,alpha2):
        model.eval()

        model.image_adapter.eval()
        model.text_adapter.eval()

        targets,features,logitses = [],[],[]

        for i, (inputs, target) in enumerate(loader):
            inputs, target = inputs.to(self._device), target.to(self._device)

            with torch.no_grad():
                outputs = model.inference(inputs, target=None, p_target=None)
                test_features = outputs['features']
                targets.append(target)
                features.append(test_features)
                logitses.append(outputs['logits'])

        targets = torch.cat(targets)
        features = torch.cat(features)
        logitses = torch.cat(logitses)
        cache_logits,_,_ = model.image_adapter(features,
                                     beta=beta,
                                     cache_values=cache_values,
                                     pow_weight=self.args['iw'])
        features = features.to(torch.float32)
        clip_logits = 100. * distance(features,model.text_adapter(clip_weights),self.args['distance'])

        sum_logits = clip_logits * alpha2 + cache_logits * alpha1
        sum_logits = sum_logits.half()
        cluster_logits = sum_logits + logitses
        acc = cls_acc(cluster_logits,targets)

        return acc

