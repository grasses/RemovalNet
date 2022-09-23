#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/08/31, homeway'


import os, pytz
import argparse
import os.path as osp
import copy
import logging
import datetime
import torch
import random
import torch.nn.functional as F
from attack import ops as FF
from torch import optim
import benchmark
from attack.finetuner import Finetuner
from utils import metric, vis as exp_F
format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S"))
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class DeepRemoval(Finetuner):
    def __init__(self, target_model, model_config, train_loader, test_loader):
        self.teacher = target_model
        self.model = copy.deepcopy(target_model)
        if type(target_model) == benchmark.ModelWrapper:
            self.model = self.model.torch_model
            self.teacher = self.teacher.torch_model
        super().__init__(model_config, model=self.model, teacher=self.teacher, train_loader=train_loader, test_loader=test_loader, init_models=False)

        """
        Alias:
        self.teacher = target model in paper
        self.model = attack model in paper
        """

        # load logger
        self.logger = logging.getLogger('RemovalNet')
        self.device = model_config.device
        self.torch_model_path = model_config.models_dir
        self.output_dir = model_config.output_dir

    def init_models(self):
        pass


    def save_torch_model(self, torch_model, step=0):
        if not osp.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        ckpt_path = osp.join(self.torch_model_path, f'final_ckpt_t{step}.pth')
        if step == 0:
            ckpt_path = osp.join(self.torch_model_path, f'final_ckpt.pth')
        torch.save(
            {'state_dict': torch_model.cpu().state_dict()},
            ckpt_path,
        )
        torch_model.to(self.device)
        self.logger.info(f"-> save model to: {self.torch_model_path}")

    def compute_grad_cost(self, y_prime, y, grad, ydist):
        assert ydist in ["l2", "kl", "cosine"]
        u = torch.matmul(grad.t(), y_prime)
        u = u / u.norm()
        v = torch.matmul(grad.t(), y)
        v = v / v.norm()
        if ydist == "l2":
            cost = (u - v).norm() ** 2
        elif ydist == "kl":
            cost = F.kl_div(u, v, reduction='sum')
        elif ydist == "cosine":
            cost = 1.0 / F.cosine_similarity(u, v, dim=0)
        else:
            raise NotImplementedError(f"-> method:{ydist} not implemented!")
        return cost

    def compute_grad(self, target_model, x, layer_idx=-2):
        grad = []
        z_a = target_model(x)
        nlls = -F.log_softmax(z_a, dim=1).mean(dim=0)
        for k in range(z_a.shape[1]):
            nll_k = nlls[k]
            _params = [p for p in target_model.parameters()]
            grads, *_ = torch.autograd.grad(nll_k, _params[layer_idx], retain_graph=True)
            grad.append(grads.flatten().clone())
        grad = torch.stack(grad).to(self.device)
        return grad

    def gradient_poison(self, target_model, x, lr=1.0, alpha=0.01, layer_idx=-4, ydist="l2"):
        """
        :param model_t0:
        :param x: torch.Tensor, input
        :param z: torch.Tensor, logit
        :return: torch.Tensor, perturbated y_prime
        """
        z_t_prime = []
        z_t = target_model(x)
        p_t = F.softmax(z_t, dim=1).detach()
        y_t = p_t.argmax(dim=1).detach()

        target_model.eval()
        for i in range(x.shape[0]):
            # find perturbation delta_i
            x_i = x[i].unsqueeze(0)
            p_t_i = p_t[i].clone()

            # obtain layer grad
            layer_grad = self.compute_grad(target_model, x=x_i, layer_idx=layer_idx).detach()
            z_prime_i = z_t[i].unsqueeze(0).clone()
            z_prime_i += torch.empty_like(z_prime_i).uniform_(-1.0, 1.0)
            for step in range(20):
                z_prime_i = z_prime_i.detach()
                z_prime_i.requires_grad = True
                p_prime_i = F.softmax(z_prime_i, dim=1)
                cost_dist = self.compute_grad_cost(p_prime_i[0], p_t_i, layer_grad, ydist=ydist)
                cost_ce = F.cross_entropy(z_prime_i, y_t[[i]])
                cost = alpha * cost_ce - (1-alpha) * cost_dist
                grad = torch.autograd.grad(cost, [z_prime_i], retain_graph=False, create_graph=False)[0]
                z_prime_i = z_prime_i - lr * grad.sign()
                '''
                adv_prob = torch.topk(F.softmax(z_i_prime, dim=1), k=3)[0]
                ben_prob = torch.topk(p_t_i, k=3)[0]
                print(
                    f"\n-> step:{step} ce:{cost_ce.item()} dist:{cost_dist.item()} "
                    f"adv_y:{z_i_prime.argmax(dim=1).item()} ben_y:{y_t[[i]].item()} norm:{(z_i_prime - z_t[i]).norm(p=2).sum().item()}")
                print(f"-> step:{step} ben_top3:{ben_prob.data} adv_top3:{adv_prob.data}\n")
                '''
            adv_prob = torch.topk(F.softmax(z_prime_i, dim=1), k=3)
            ben_prob = torch.topk(p_t_i, k=3)
            z_t_prime.append(z_prime_i)
        z_t_prime = torch.cat(z_t_prime)
        mse_dist = F.mse_loss(z_t_prime, z_t)
        print(
            f"-> gradient_poison() ce_loss:{cost_ce.item()} {ydist}_dist:{cost_dist.item()} "
            f"adv_y:{z_prime_i.argmax(dim=1).item()} ben_y:{y_t[[i]].item()} mse_dist:{mse_dist}")
        print(f"-> gradient_poison() ben_top3:{ben_prob[0].cpu().data}_{ben_prob[1].cpu().data} adv_top3:{adv_prob[0].cpu().data}_{adv_prob[1].cpu().data}")
        return z_t_prime.detach()

    def feature_poison(self, target_model, x, y, layer_index=5, lr=0.1, alpha=0.5, steps=20, ydist="l2"):
        z = target_model.mid_forward(x, layer_index=layer_index).detach()
        z_prime = z.clone() + torch.empty_like(z).uniform_(-10, 10)
        for step in range(steps):
            z_prime = z_prime.detach()
            z_prime.requires_grad = True
            logit_t = target_model.bak_forward(z_prime, layer_index=layer_index)

            dist_mse = F.mse_loss(z_prime, z)
            if ydist == "l2":
                u = z_prime / z_prime.norm()
                v = z / z.norm()
                loss_dist = (u - v).norm()
            elif ydist == "cosine":
                loss_dist = F.cosine_similarity(z_prime.view(len(x), -1), z.view(len(x), -1), dim=1).mean()
            elif ydist == "kl":
                loss_dist = dist_mse
            # min CELoss & max cosine dist
            loss_ce = F.cross_entropy(logit_t, y)
            loss = alpha * loss_ce + (1 - alpha) * loss_dist
            grad = torch.autograd.grad(loss, [z_prime], retain_graph=False, create_graph=False)[0]
            z_prime = z_prime.detach() - lr * grad.sign()
        print(f"-> feature_poison() ce_loss:{loss.item()} {ydist}_dist:{loss_dist.item()}")
        return z_prime.detach()

    def model_poison(self):
        """from Byzantine methods"""
        pass

    def maximize_deviation(self, teacher, student, x, logits):
        z_prime, y_prime = [], []
        y_t = logits.argmax(dim=1).detach()

        # find perturbated latent logit_prime
        #z_prime = self.feature_poison(model_t0, x, y_t)

        # find perturbated logit y_prime
        for i in range(x.shape[0]):
            x_i = x[i].unsqueeze(0)
            y_i = y_t[[i]]
            delta_i, objval, sobjval = self.logits_poison(teacher, student, x_i, y_i)
            y_prime_i = y_i + delta_i
            y_prime.append(y_prime_i)
        y_prime = torch.stack(y_prime).detach()
        return z_prime, y_prime

    def train(self):
        file_path = os.path.join(self.output_dir, f"Removalnet_{self.args.ydist}")
        learning_data = {
            "t": [],
            "acc": [],
            "loss_at": [],
            "loss_kd": [],
            "keys": ["acc", "loss_at", "loss_kd"]
        }

        teacher = self.teacher
        student = self.model

        teacher.eval()
        teacher = teacher.to(self.args.device)
        student.train()
        student = student.to(self.args.device)

        train_loader = self.train_loader
        test_loader = self.test_loader

        args = self.args
        alpha = args.steal_alpha
        T = args.temperature
        iterations = self.args.iterations + 1

        lr = 1e-3
        optimizer = optim.SGD(
            student.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            iterations,
        )
        args.test_interval = 20
        dataloader_iterator = iter(train_loader)
        for t in range(iterations):
            # load next batch data
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            x, y = batch.to(self.device), label.to(self.device)

            layer_index = random.randint(2, 5)
            # training process
            logit_t = teacher(x).detach()
            labels = logit_t.argmax(dim=1)
            #feats_prime = self.feature_poison(student, x=x, y=labels, layer_index=layer_index, ydist=args.ydist).detach()
            logit_prime = self.gradient_poison(student, x, ydist=args.ydist).detach()

            student.train()
            teacher.eval()
            optimizer.zero_grad()

            # maximize label-level deviation
            logit_s = student(x)
            loss_kd = FF.loss_kd(logit_s, labels, logit_prime, alpha=alpha, T=T)

            # maximize feature-level deviation
            '''
            feats_z = student.mid_forward(x, layer_index=layer_index)
            loss_at = (lr * 1e4) * FF.loss_at(feats_prime, feats_z)
            loss = loss_kd + loss_at
            '''

            # maximize feature space deviation
            layer_index = random.randint(2, 5)
            z_s = student.mid_forward(x, layer_index=layer_index)
            z_t = teacher.mid_forward(x, layer_index=layer_index).detach()
            loss_at = FF.loss_at(z_s, z_t)
            loss = loss_kd - (lr * 1e6) * loss_at

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # log preview && save model
            print(f"-> train() step:{t} layer_index:{layer_index} loss:{loss.item()} loss_at:{loss_at.item()} loss_kd:{loss_kd.item()}")
            if (t > 0) and ((t % args.test_interval == 0) or (t == iterations - 1)):
                acc1, acc2 = exp_F.plot_logist_embedding(teacher, student, test_loader, out_root=self.output_dir, file_name=f"RemovalNet_{args.ydist}_{t}.pdf")
                print(f"-> t:{t} teacher:{acc1}% student:{acc2}% loss:{loss.item()} loss_at:{loss_at.item()} loss_kd:{loss_kd.item()}")
                self.save_torch_model(student.cpu(), step=t)

            # testing & exp visualization
            _best_topk_acc, topk_acc, test_loss = metric.topk_test(student, test_loader, device=self.device, epoch=t, debug=True)
            learning_data["t"].append(t)
            learning_data["acc"].append(topk_acc["top-1"])
            learning_data["loss_at"].append(FF.numpy(loss_at))
            learning_data["loss_kd"].append(FF.numpy(loss_kd))
            exp_F.plot_learning_curve(learning_data, file_path=file_path)
            self.save_torch_model(student.cpu())
            print()


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="pretrain(resnet18,ImageNet)-",
                        required=True, help="model 1.")
    parser.add_argument("-model2", action="store", dest="model2", default="pretrain(resnet18,ImageNet)-",
                        required=False, help="model 1.")

    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-ydist", action="store", default="l2", type=str, choices=["l2", "cosine", "kl"], help="distance of adv logits")

    parser.add_argument("-batch_size", action="store", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=999, type=int, help="Default seed of numpy/pyTorch")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT
    args.namespace = format_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.removal_root = osp.join(args.out_root, "RemovalNet")
    return args


def main():
    import benchmark
    from dataset import loader

    args = get_args()
    filename = str(osp.basename(__file__)).split(".")[0]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                        filename=f"{args.logs_root}/{filename}_{args.namespace}.txt")
    bench = benchmark.ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    model1 = bench.get_model_wrapper(args.model1)

    train_loader = loader.get_dataloader(model1.dataset_id, split="train", batch_size=args.batch_size)
    test_loader = loader.get_dataloader(model1.dataset_id, split="test", batch_size=500)

    if "quantize" in str(model1):
        args.device = torch.device("cpu")

    model_args = benchmark.model_args()
    model_args.layer_index = 5
    model_args.steal_alpha = 0.5
    model_args.temperature = 1
    model_args.ydist = args.ydist


    removalnet = f"stealthnet({model_args.steal_alpha},{model_args.temperature},{args.ydist})-"
    model_name = f"{str(model1)}{removalnet}"
    out_root = osp.join(ROOT, "output", f"Removal/{model_name}")
    model_args.output_dir = out_root
    if not osp.exists(out_root):
        os.makedirs(out_root)

    model_args.models_dir = osp.join(args.models_dir, f'{model_name}')
    removal = DeepRemoval(model1, model_args, train_loader=train_loader, test_loader=test_loader)
    removal.train()


if __name__ == "__main__":
    main()

    """
        Example command:
        <===========================  Flower102-resnet18:20220901_Test  ===========================>
        SCRIPT="attack.deepremoval"
        python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -model2 "" -device 2
    """


















