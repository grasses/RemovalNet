#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import copy
import logging
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b
from torch.autograd import Variable
from benchmark import ImageBenchmark
from utils import helper
from defense.modeldiff import ModelDiff
from exp import vis as exp_F
sys_args = helper.get_args()
filename = str(osp.basename(__file__)).split(".")[0]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-8s \033[1;35m %(message)s \033[0m")


class FeatureAdversary(object):
    def __init__(self, model, x0, feat_k_guide, label_k_guide, alpha=0.4):
        self.device = sys_args.device
        self.input_shape = [1, 3, 224, 224]
        self.model = model
        self.alpha = alpha
        self.x0 = x0.view(self.input_shape).to(self.device)
        self.feat_k_guide = feat_k_guide.view(-1).to(self.device)
        self.label_k_guide = label_k_guide
        self.loss_value = None
        self.grads_values = None
        self.iter = 0

    def loss(self, x):
        def record_act(self, input, output):
            self.out = output
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module1 = module
        module1.register_forward_hook(record_act)

        Ix = Variable(torch.from_numpy(x.reshape(self.input_shape)).to(self.device).float())
        Ix.requires_grad = True

        output = self.model(Ix)
        feat_x = module1.out.view(-1).to(self.device)
        loss = (1-self.alpha) * torch.dist(feat_x, self.feat_k_guide, p=2) + self.alpha * torch.dist(Ix.view(-1), self.x0.view(-1), p=2)
        loss.backward(retain_graph=True)

        self.loss_value = loss
        self.grad_values = Ix.grad.cpu().numpy().flatten().astype(float)

        if (self.iter % 200 == 0):
            pred_result = output.argmax(1, keepdim=True).item()
            #print(f"-> iter:{self.iter}, loss:{self.loss_value}, guide_label:{self.label_k_guide} predict_label:{pred_result}")
        self.iter += 1
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


class StealthNet:
    def __init__(self, bench, model1, model2, alpha=0.6, iter_size=0):
        self.alpha = alpha
        self.iter_size = iter_size
        self.device = sys_args.device
        self.logger = logging.getLogger('StealthNet')

        self.model1 = model1
        self.model2 = model2
        self.source_name = str(model1)
        self.attack_name = str(model2)
        self.source_model = model1.torch_model.to(self.device)
        self.origin_model = model2.torch_model.to(self.device)
        self.attack_model = copy.deepcopy(model2.torch_model).to(self.device)
        self.train_loader = bench.get_dataloader("Flower102", split='train')
        self.test_loader = bench.get_dataloader("Flower102", split='test')

        self.scope_name = f"{model2}stealthnet({self.alpha},{self.iter_size})-"
        self.torch_model_path = osp.join(bench.models_dir, self.scope_name)

        """
        trainset_dict = {}
        for x, y in self.train_loader:
            for idx, _y in enumerate(y):
                if int(_y) not in trainset_dict.keys():
                    trainset_dict[int(_y)] = []
                trainset_dict[int(_y)].append(x[idx])
        for k, v in trainset_dict.items():
            trainset_dict[k] = torch.cat(trainset_dict[k])
        self.trainset = trainset_dict
        """

    def save_torch_model(self, torch_model):
        if not osp.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        ckpt_path = osp.join(self.torch_model_path, 'final_ckpt.pth')
        torch.save(
            {'state_dict': torch_model.cpu().state_dict()},
            ckpt_path,
        )
        torch_model.to(self.device)
        self.logger.info(f"-> save model to: {self.torch_model_path}")

    @staticmethod
    def eval(model, test_loader, device=sys_args.device, epoch=0, debug=True):
        test_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = F.cross_entropy(output, y)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= (1.0 * len(test_loader.dataset))
        acc = 100.0 * correct / len(test_loader.dataset)
        msg = "-> For E{:d}, [Test] loss={:.5f}, acc={:.3f}%".format(
            int(epoch),
            test_loss,
            acc
        )
        if debug:
            print(msg)
        return acc, test_loss

    def eval_similarity(self, model1, model2, profiling_inputs):
        comparison = ModelDiff(model1, model2)
        comparison.none_optimized_compare(profiling_inputs)

    def compute_fingerprint_samples(self, model, origin_model, x, y, eps=10/255):
        attack_model = copy.deepcopy(model)
        def record_act(self, input, output):
            self.out = output

        def random_second_likely_idx(y, logits, iter_size):
            cnt = 0
            predict_y = logits.argmax(dim=1).view(-1)
            second_likely_y = torch.topk(logits, k=2, dim=1)[1][:, 1]
            for idx_sur in range(len(logits)):
                if cnt > iter_size:
                    break
                idx_list = ((predict_y[idx_sur] == second_likely_y[idx_sur]).nonzero(as_tuple=True)[0])
                if len(idx_list) == 0:
                    continue
                idx_gud = idx_list[0]
                cnt += 1
                yield idx_sur, idx_gud

        def random_idx(y, logits, iter_size):
            for i in range(iter_size):
                idx_sur, idx_gud = np.random.randint(0, len(y), size=2)
                while y[idx_sur] == y[idx_gud]:
                    idx_sur, idx_gud = np.random.randint(0, len(y), size=2)
                yield idx_sur, idx_gud

        for name, module in attack_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module1 = module
        handle1 = module1.register_forward_hook(record_act)

        adv_x = []
        attack_model = attack_model.eval()
        for idx_sur, idx_gud in random_idx(y, attack_model(x), iter_size=self.iter_size):
            attack_model(x)
            feat_k_guide = copy.deepcopy(module1.out[idx_gud].detach().cpu())
            FA = FeatureAdversary(attack_model, x0=x[idx_sur].clone().to(self.device), feat_k_guide=feat_k_guide, label_k_guide=y[idx_gud])
            tmp_x, min_val, info = fmin_l_bfgs_b(func=FA.loss,
                                                 x0=(x[idx_sur].clone().cpu()).flatten(),
                                                 # bounds=constraint
                                                 fprime=FA.grads,
                                                 maxiter=1000)
            tmp_x = torch.from_numpy(tmp_x.reshape([1, 3, 224, 224])).float().to(self.device)
            output = attack_model(tmp_x)
            pred_result = output.argmax(1, keepdim=True).item()
            adv_x.append(tmp_x.cpu().detach())
            self.logger.info(f"-> source_label:{y[idx_sur]} guide_label:{y[idx_gud]} predict_label:{pred_result}")
            handle1.remove()
            del module1.out, FA
        adv_x = torch.cat(adv_x, dim=0).to(self.device)
        adv_y = attack_model(adv_x).argmax(dim=1)
        attack_model.train()
        return adv_x, adv_y

    def fingerprint_unlearning(self, epochs=10):
        self.origin_model.eval()
        self.attack_model.train()
        optimizer = torch.optim.SGD(
            self.attack_model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=5e-3,
        )
        self.eval(self.attack_model, self.test_loader, epoch=0)
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss_ce, _, loss_sim = self.compute_loss(self.attack_model, self.origin_model, x=x, y=y)
                loss = self.alpha * loss_sim + (1-self.alpha) * loss_ce
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.logger.info(f"-> E{epoch} [{step}/{len(self.train_loader)}] loss:{loss.item()}")
                self.eval(self.attack_model, self.test_loader, epoch=epoch)
                self.save_torch_model(self.attack_model)
                print()
            exp_F.plot_logist_embedding(self.attack_model, self.origin_model, self.test_loader, file_name=f"{self.scope_name}_e{epoch}.pdf")
            print()
            print()


    def fingerprint_unlearning_ben_adv(self, epochs=10):
        self.origin_model.eval()
        self.attack_model.train()
        optimizer = torch.optim.SGD(
            self.attack_model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=5e-3,
        )
        self.eval(self.attack_model, self.test_loader, epoch=0)

        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                loss_ce, _, loss_sim = self.compute_loss(self.attack_model, self.origin_model, x=x, y=y)
                adv_x, adv_y = self.compute_fingerprint_samples(self.attack_model, self.origin_model, x=x, y=y)
                _, adv_loss_ce, adv_loss_sim = self.compute_loss(self.attack_model, self.origin_model, x=adv_x, y=adv_y)
                loss = (adv_loss_sim + loss_sim) * self.alpha + loss_ce * (1 - self.alpha)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.logger.info(f"-> E{epoch} [{step}/{len(self.train_loader)}] loss:{loss.item()} loss_ce:{loss_ce.item()} loss_sim:{adv_loss_sim.item()}")
                self.eval(self.attack_model, self.test_loader, epoch=epoch)
                self.save_torch_model(self.attack_model)
                print()
            del adv_x, adv_y
            print()
            print()

    def compute_loss(self, attack_model, origin_model, x, y):
        # get last layer feature maps
        def record_act(self, input, output):
            self.out = output
        for name, module in origin_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module1 = module
        for name, module in attack_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module2 = module
        module1.register_forward_hook(record_act)
        module2.register_forward_hook(record_act)
        x, y = x.to(self.device), y.to(self.device)

        out1 = origin_model(x).detach()
        out2 = attack_model(x)
        feature1 = module1.out.view(y.size(0), -1).detach()
        feature2 = module2.out.view(y.size(0), -1)

        loss_ce = F.cross_entropy(out2, y)
        loss_guide_ce = F.cross_entropy(out2, out1.argmax(dim=1).view_as(y))
        #loss_guide_sim = 10.0 / torch.dist(feature1.view(-1), feature2.view(-1), p=2).sum()
        loss_guide_sim = torch.nn.CosineSimilarity(dim=1)(feature1, feature2).sum() / y.size(0)
        self.logger.info(f"-> loss_ce:{loss_ce} loss_guide_ce:{loss_guide_ce} loss_guide_sim:{loss_guide_sim}")
        del module1.out, module2.out, feature1, feature2
        return loss_ce, loss_guide_ce, loss_guide_sim


def main():
    args = helper.get_args()
    benchmark = ImageBenchmark(
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir
    )
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)
    net = StealthNet(benchmark, model1=model1, model2=model2)
    net.fingerprint_unlearning()

if __name__ == "__main__":
    main()