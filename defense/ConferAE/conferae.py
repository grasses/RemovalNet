#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/25, homeway'


"""
code for: 
    Deep Neural Network Fingerprinting by Conferrable Adversarial Examples
"""

import sys
import time
import argparse
import os, datetime, pytz
import os.path as osp
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ops, cfg, Trainer
from model.inputx224 import torchvision_models
from defense import Fingerprinting
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ConferAE(Fingerprinting):
    def __init__(self, model1, model2, out_root, device,
                 tau=0.9, batch_size=128):
        super().__init__(model1, model2, device=device, out_root=out_root)

        # init logger
        self.logger = logging.getLogger('ConferAE')
        self.logger.info(f'-> comparing {model1} vs {model2}')

        # init dataset
        self.tau = tau
        self.device = torch.device("cpu")
        self.dataset = model1.dataset_id
        ops.set_default_seed(100)
        self.train_loader = model1.benchmark.get_dataloader(self.dataset, split='train', batch_size=batch_size)
        self.test_loader = model1.benchmark.get_dataloader(self.dataset, split='test', batch_size=200)
        self.finger_loader = model1.benchmark.get_dataloader(self.dataset, split='test', shuffle=True, batch_size=1)

        self.bounds = self.test_loader.bounds
        self.num_classes = self.train_loader.dataset.num_classes

        # init model
        self.arch1 = str(model1)
        self.arch2 = str(model2)
        self.model1 = model1.torch_model.to(self.device)
        self.model2 = model2.torch_model.to(self.device)
        self.surr_list = []
        self.ref_list = []
        ops.set_default_seed(100)

    def gen_models(self, input_size=224):
        arch_list = [
            "resnet18", "resnet34", "vgg11", "vgg16", #"alexnet"
        ]
        lr_list = [
            1e-2, 8e-3, 5e-3
        ]

        # generate surrogate models
        args = cfg.model_args()
        for step, arch_id in enumerate(arch_list):
            for lr in lr_list:
                ops.set_default_seed(int(time.time()))
                model_str = f'surr_pretrain({arch_id},ImageNet)-steal({arch_id},{self.dataset},{lr})-'
                self.logger.info(f"-> train surrogate model: {model_str}")
                student = eval(f"torchvision_models.{arch_id}(num_classes={self.num_classes}, pretrained='imagenet')")
                args.network = arch_id
                args.steal = True
                args.reinit = True
                args.steal_alpha = 1
                args.temperature = 1
                args.lr = lr
                args.weight_decay = 5e-3
                args.momentum = 0.9
                torch_model_path = os.path.join(self.ckpt_root, model_str)
                args.output_dir = torch_model_path
                trainer = Trainer(
                    args,
                    student=student, teacher=self.model1,
                    train_loader=self.train_loader, test_loader=self.test_loader,
                    torch_model_path=torch_model_path,
                    seed=int(time.time())
                )
                student = trainer.load_torch_model()
                self.surr_list.append(student.to("cpu").eval())

        # generate reference models
        args = cfg.model_args()
        for step, arch_id in enumerate(arch_list):
            for lr in lr_list:
                ops.set_default_seed(int(time.time()))
                model_str = f'ref_pretrain({arch_id},ImageNet)-transfer({arch_id},{self.dataset},{lr})-'
                self.logger.info(f"-> train reference model: {model_str}")
                dataset_id = self.dataset
                student = eval(f"torchvision_models.{arch_id}(num_classes={self.num_classes}, pretrained='imagenet')")
                args.network = arch_id
                args.ft_ratio = 0.8
                args.reinit = True
                args.lr = lr
                args.weight_decay = 5e-3
                args.momentum = 0.9
                torch_model_path = os.path.join(self.ckpt_root, model_str)
                args.output_dir = torch_model_path
                trainer = Trainer(
                    args,
                    student=student, teacher=self.model1,
                    train_loader=self.train_loader, test_loader=self.test_loader,
                    torch_model_path=torch_model_path,
                    seed=int(time.time())
                )
                student = trainer.load_torch_model()
                self.ref_list.append(student.to("cpu").eval())

        print(f"-> size(surrogate):{len(self.surr_list)} size(reference):{len(self.ref_list)}")
        return self.surr_list, self.ref_list

    def extract(self, **kwargs):
        ops.set_default_seed(100)
        path = osp.join(self.fingerprint_root, f"{self.arch1}_{self.dataset}.pt")
        if osp.exists(path):
            self.logger.info(f"-> load fingerprint from: {path}\n")
            return torch.load(path, map_location="cpu")

        self.logger.info("-> gen surrogate & reference models\n")
        surr_list, ref_list = self.gen_models()

        self.logger.info("-> gen fingerprint")
        test_x, test_y, conf_list = [], [], []
        for step, (x, y) in enumerate(self.finger_loader):
            x, y = x.to(self.device), y.to(self.device)
            _test_x, _test_y, conf = self.optimize(M=self.model1, Surr=surr_list, Ref=ref_list, x=x, y=y, tau=self.tau)
            if len(_test_x) > 0:
                test_x.append(_test_x)
                test_y.append(_test_y)
                conf_list.append(float(conf))
                torch.save([test_x, test_y, conf_list], path)

        fingerprint = {
            "test_x": torch.cat(test_x),
            "test_y": torch.cat(test_y),
            "conf": torch.tensor(conf_list).float()
        }
        self.logger.info(f"-> save fingerprint to: {path}")
        torch.save(fingerprint, path)
        return fingerprint


    def verify(self, fingerprint, **kwargs):
        # verify using CAEAcc
        pass


    def compare(self, **kwargs):
        pass

    def cal_transferability(self, models, x, t):
        """
        Equ. (1) targeted transferability for a class t and a set of models.
        :param models:
        :param x:
        :param t:
        :return:
        """
        x, t = x.to(self.device), t.to(self.device)
        correct = torch.zeros(t.size())
        preds, confs = [], []
        for idx, model in enumerate(models):
            model.eval()
            model.to(self.device)
            pred_y = model(x).argmax(dim=1)
            preds.append(int(pred_y[0]))
            confs.append(float(torch.max(F.softmax(model(x), dim=1), dim=1)[0][0]))
            correct += torch.gt(pred_y == t, 0).int()
        return (correct.float() / len(models)).float(), preds, confs

    def cal_conferrability(self, Surr, Ref, x, t, debug=False):
        """
        Equ. (2) x’s transferability to surrogate and reference models.
        :param Surr:
        :param Ref:
        :param x:
        :param t:
        :return:
        """
        trans_surr, preds_surr, confs_surr = self.cal_transferability(Surr, x, t)
        trans_ref, preds_ref, confs_ref = self.cal_transferability(Ref, x, t)
        if debug:
            print(f"-> CONF trans_surr:{trans_surr} trans_ref:{trans_ref}\n-> CONF preds_surr:{preds_surr} preds_ref:{preds_ref}\n-> CONF confs_surr:{confs_surr} confs_ref:{confs_ref}")
        scores = torch.mul(trans_surr, (1-trans_ref))
        return scores,

    def cal_conferrable_ensemble(self, models, x, p=0.3, softmax=True):
        """
        Equ. (3)/(4), input surrogate models & reference models, return ensemble predictions.
        :param model_list:
        :param x:
        :param p:
        :param softmax:
        :return:
        """
        conf, pred =[] , []
        x = x.to(self.device)
        for idx, model in enumerate(models):
            model.eval()
            model.to(self.device)
            z = nn.Dropout2d(p=p)(model(x))
            if softmax:
                # softmax process, give up F.softmax() to avoid "modified by an inplace operation" error
                maxes = torch.max(z, 1, keepdim=True)[0]
                x_exp = torch.exp(z - maxes)
                x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
                z = x_exp / x_exp_sum
            pred.append(int(z.argmax(dim=1)[0]))
            conf.append(float(torch.max(z, dim=1)[0][0]))

            if idx == 0:
                logit = z
            else:
                logit += z
        logit /= len(models)
        if not softmax:
            return F.softmax(logit, dim=1), pred, conf
        return logit, pred, conf


    def cal_conferrable_ensemble_method(self, x, Surr, Ref):
        """
        Equ. (5) Conferrable Ensemble Method (CEM)
        :param x: Torch.tensor, x
        :param Surr: list, surrogate models of M
        :param Ref: list, reference model
        :return: sigmoid logit
        """
        Softmax = nn.Softmax(dim=1)
        logit_surr, preds_surr, confs_surr = self.cal_conferrable_ensemble(models=Surr, x=x, softmax=True)
        logit_ref, preds_ref, confs_ref = self.cal_conferrable_ensemble(models=Ref, x=x, softmax=True)
        logit_reverse_ref = torch.ones(logit_ref.shape, requires_grad=True) - logit_ref
        logit_ens = Softmax(torch.mul(logit_surr, logit_reverse_ref))
        return logit_surr, logit_ref, logit_ens


    def optimize(self, M, Surr, Ref, x, y, tau, eps=0.2, steps=1000, lr=30, alpha=1.0, beta=0.1, gama=0.1):
        CEloss = nn.CrossEntropyLoss()
        M.eval()
        M.to(self.device)
        x_0 = x.clone().detach().to(self.device)

        # random select a target label t
        '''
        num_classes = self.train_loader.dataset.num_classes
        ll = list(range(num_classes))
        ll.remove(int(y))
        t = torch.tensor([random.choice(ll)]).long().to(self.device)
        '''
        delta = torch.empty_like(x).uniform_(-0.5, 0.5).to(self.device)
        optimizer = torch.optim.SGD([delta], lr=lr, weight_decay=1e-4)
        for step in range(steps):
            delta.requires_grad = True
            optimizer.zero_grad()

            x_prime = x_0 + delta
            logit_surr, preds_surr, confs_surr = self.cal_conferrable_ensemble(models=Surr, x=x_prime, softmax=True)
            logit_ref, preds_ref, confs_ref = self.cal_conferrable_ensemble(models=Ref, x=x_prime, softmax=True)
            logit_reverse_ref = torch.ones(logit_ref.shape, requires_grad=True) - logit_ref
            logit_ens = F.softmax(torch.mul(logit_surr, logit_reverse_ref), dim=1)

            z_x = M(x_0)
            z_x_prime = M(x_prime)
            if step == 0:
                t = logit_ens.argmax(dim=1).detach()
                if t[0].item() == z_x.argmax(dim=1)[0].item():
                    break

            # term1: +α x H(1, max_t[ME(x′)_t])
            loss1 = CEloss(logit_ens, t)
            # term2: -β x H(M(x_0), M(x′))
            #loss2 = -CEloss(z_x_prime, z_x.argmax(dim=1))
            loss2 = CEloss(z_x_prime, t)
            # term3: +γ x H(M(x′), Surr(S_M , x′))
            loss3 = CEloss(logit_surr, z_x_prime.argmax(dim=1))
            loss = alpha * loss1 + beta * loss2 + gama * loss3

            loss.backward()
            optimizer.step()

            #grad = torch.autograd.grad(loss, [delta])[0]
            #delta = (delta - lr * grad).detach()
            #delta = torch.nn.utils.clip_grad_norm_(delta - lr * grad, max_norm=eps).detach()

            # compute only for log
            debug = True if step % 20 == 0 else False
            y_x = int(z_x.argmax(dim=1)[0])
            y_x_prime = int(z_x_prime.argmax(dim=1)[0])
            ens_y = int(logit_ens.argmax(dim=1)[0])
            surr_y = int(logit_surr.argmax(dim=1)[0])
            ref_y = int(logit_ref.argmax(dim=1)[0])
            ens_conf = round(float(logit_ens[0][ens_y]), 3)
            surr_conf = round(float(logit_surr[0][surr_y]), 3)
            ref_conf = round(float(logit_ref[0][ref_y]), 3)

            norm = round(float(torch.norm(delta, p=2)), 4)
            conf = float(self.cal_conferrability(Surr=Surr, Ref=Ref, x=(x_0 + delta).clone(), t=t, debug=debug)[0])
            print(f"-> step:{step} t:{int(t)} M(x+δ):{y_x_prime} M(x):{y_x} CEM_y:{ens_y}_c:{ens_conf} surr_y:{surr_y}_c:{surr_conf} ref_y:{ref_y}_c:{ref_conf} conf:{round(conf, 4)} loss:{loss.item()} norm(δ):{norm}")
            if debug:
                print(
                    f"-> CEM preds_surr:{preds_surr} preds_ref:{preds_ref}\n-> CEM confs_surr:{confs_surr} confs_ref:{confs_ref}")
                values, indexs = torch.topk(logit_surr.detach(), dim=1, k=5)
                print("-> CEM logit_surr", values.detach(), indexs)
                values, indexs = torch.topk(logit_ref.detach(), dim=1, k=5)
                print("-> CEM logit_ref", values.detach(), indexs)
                print("\n\n")


            if conf > tau:
                self.logger.info(f"-> conf:{conf} break iteration!!!")
                test_x = (x_0 + delta).detach()
                test_y = M(test_x).detach()
                return test_x, test_y, conf
            sys.stdout.flush()
        return [], [], []


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=False, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2",
                        default="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-", required=False, help="model 2.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-batch_size", action="store", default=200, type=int, help="GPU device id")
    parser.add_argument("-seed", default=999, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.conferae_root = osp.join(args.out_root, "ConferAE")
    return args


def main():
    from benchmark import ImageBenchmark
    args = get_args()
    filename = str(osp.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        )  # filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")

    benchmark = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)
    confer = ConferAE(model1=model1, model2=model2, out_root=args.conferae_root, device=args.device)
    confer.extract()



if __name__ == "__main__":
    main()

    """
        Example command:
        <===========================  Flower102-resnet18  ===========================>
        model1="pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.ConferAE.conferae" -model1 $model1 -model2 "train(resnet18,Flower102)-" -device 1


        <===========================  Flower102-mbnetv2  ===========================>
        model1="pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-"
        python -m "defense.ConferAE.conferae" -model1 $model1 -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-stealthnet(0.7,20)-" -device 1
    """

