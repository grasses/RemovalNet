#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2023/01/14, homeway'


import torch
import numpy as np
import sys, os, argparse, random
import os.path as osp
from tqdm import tqdm
from torch.nn import functional as F
from art.estimators.classification.pytorch import PyTorchClassifier
from benchmark import ImageBenchmark
from .knockoffnets import KnockoffNets
from dataset import loader as dloader
from model import loader as mloader
from utils import helper, metric


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", action="store", dest="model1", default="train(resnet50,CIFAR10)-", required=False, help="Benchmark model1")
    parser.add_argument("-subset", required=True, type=str, help="Substitute dataset for KnockoffNets")
    parser.add_argument("-batch_size", required=False, type=int, default=128, help="batch_size of dataloader")
    parser.add_argument("-nb_stolen", required=False, type=int, default=200, help="batch_size of dataloader")
    parser.add_argument("-batch_size_query", required=False, type=int, default=2000, help="batch_size of dataloader")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-backends", default=0, type=int, choices=[0, 1], help="CUDA backends")
    args, unknown = parser.parse_known_args()
    args.ROOT = helper.ROOT
    args.namespace = helper.curr_time
    args.out_root = osp.join(args.ROOT, "output")
    args.logs_root = osp.join(args.ROOT, "logs")
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    helper.set_default_seed(seed=args.seed)
    for path in [args.datasets_dir, args.models_dir, args.out_root, args.logs_root]:
        if not osp.exists(path):
            os.makedirs(path)
    return args


def train_knockoff_adaptive_strategy(cfg, vic_model, atk_model, train_loader, test_loader):
    '''
    from MLaaS.server.storage import Storage
    storage_root = f"output/storage"
    prefix = f"{args.vic_model}_{args.vic_data}"
    storage = Storage(storage_root=storage_root, prefix=prefix)
    blackbox = BlackBox(model=vic_model, save_process=storage.add_query, device=args.device, anchor_feats=anchor_feats)
    '''
    loss_fn = torch.nn.CrossEntropyLoss()
    vic_model.eval()
    vic_optimizer = torch.optim.Adam(vic_model.parameters(), lr=cfg.lr)
    vic_art = PyTorchClassifier(
        model=vic_model, loss=loss_fn, optimizer=vic_optimizer,
        input_shape=cfg.input_shape, nb_classes=test_loader.num_classes,
        clip_values=test_loader.bounds,
        device_type="gpu"
    )
    atk_optimizer = torch.optim.SGD(atk_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    atk_art = PyTorchClassifier(
        model=atk_model, loss=loss_fn, optimizer=atk_optimizer,
        input_shape=cfg.input_shape, nb_classes=test_loader.num_classes,
        clip_values=train_loader.bounds,
        device_type="gpu"
    )
    knockoff = KnockoffNets(
        classifier=vic_art,
        batch_size_fit=cfg.batch_size,
        batch_size_query=cfg.batch_size_query,
        nb_epochs=1,
        nb_stolen=cfg.nb_stolen,
        nb_classes=test_loader.num_classes,
        sampling_strategy="adaptive",
        reward="all",
        verbose=False
    )

    step = 0
    best_acc = 0.0
    sampel_cnt = 0
    curr_budget = 0
    phar = tqdm(range(cfg.query_budget + 1000))
    loader = iter(train_loader)
    while True:
        try:
            batch_x, batch_y = loader.next()
            x = batch_x.numpy()
            pred_y = vic_model(batch_x.to(cfg.device)).detach().cpu().argmax(dim=1)
            y = F.one_hot(pred_y, num_classes=test_loader.num_classes).numpy()
            (selected_idxs, selected_x, selected_y), probs, thieved_ptc = \
                knockoff.extract(x=x, y=y, thieved_classifier=atk_art, use_global=False)

            thieved_logits = thieved_ptc.predict(x=x)
            thieved_preds = np.argmax(thieved_logits, axis=1)
            victim_preds = np.argmax(vic_art.predict(x=x), axis=1)

            curr_budget += len(selected_idxs)
            phar.update(len(selected_idxs))
            train_correct = np.sum(victim_preds == thieved_preds)
            train_acc = 100.0 * train_correct / len(victim_preds)
            train_loss = F.cross_entropy(torch.from_numpy(thieved_logits), torch.from_numpy(victim_preds).long())
            phar.set_description(f'-> Step:{step} B:[{curr_budget}/{cfg.query_budget}] Loss:{train_loss} | Acc: {train_acc}% ({train_correct}/{len(y)})')

            # epoch = _epoch * tick_break + int(step/tick_step) + 1
            if (step < 1000 and step % 100 == 0) or step % 1000 == 0:
                best_topk_acc, topk_acc, test_loss = metric.topk_test(thieved_ptc.model, test_loader, cfg.device, epoch=step, debug=True)
                if topk_acc["top1"] > best_acc:
                    best_acc = topk_acc["top1"]
                ckpt_path = osp.join(cfg.models_dir, f"{cfg.model2}/final_ckpt_s{cfg.seed}_t{step}.pth")
                torch.save(
                    {
                        'top1_acc': topk_acc["top1"],
                        'top3_acc': topk_acc["top3"],
                        'top5_acc': topk_acc["top5"],
                        'iters': step,
                        'seed': cfg.seed,
                        'state_dict': atk_model.cpu().state_dict()
                    },
                    ckpt_path,
                )
                atk_model.to(cfg.device)
                print(f"-> For E{step} [Test] cnt:{sampel_cnt} test_acc:{topk_acc['top1']:.3f}%")
                print(f"-> Best Top-1={best_topk_acc['top1']} Top-3={best_topk_acc['top3']} Top-5={best_topk_acc['top5']}")
                print()
            step += 1
            sampel_cnt += len(selected_x)
            sys.stdout.flush()
            if curr_budget > cfg.query_budget + 1000:
                break
        except Exception as e:
            print("-> find error!!!", e)
            loader = iter(train_loader)


def train_knockoff_random_strategy(cfg, vic_model, atk_model, train_loader, test_loader):
    vic_model.eval()
    vic_model.to(cfg.device)
    atk_model.train()
    atk_model.to(cfg.device)
    optimizer = torch.optim.SGD(atk_model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    train_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    count = 0
    best_acc = 0.0
    iter_loader = iter(train_loader)
    query_size = int(cfg.query_budget / cfg.nb_stolen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        query_size,
    )
    phar = tqdm(range(query_size))
    for step in phar:
        try:
            inputs, targets = next(iter_loader)
        except:
            iter_loader = iter(train_loader)
            inputs, targets = next(iter_loader)

        if cfg.backends:
            atk_model.cuda()
            vic_model.cuda()
            inputs = inputs.cuda()
            targets = vic_model(inputs).argmax(dim=1).detach().cuda()
        else:
            atk_model.to(cfg.device)
            inputs = inputs.to(cfg.device)
            targets = vic_model(inputs).argmax(dim=1).detach()

        optimizer.zero_grad()
        outputs = atk_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        count += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        phar.set_description(
            '-> Step: {:d} Seed:{:d} | Loss: {:.3f} | Acc:{:.3f}% Best:{:.3f} ({:d}/{:d}) | Lr: {:.6f}'.format(
                step, cfg.seed, train_loss / count, 100. * correct / count,
                best_acc, correct, count,
                float(current_lr)
            )
        )

        if (step < 1000 and step % 100 == 0) or step % 1000 == 0:
            best_topk_acc, topk_acc, test_loss = metric.topk_test(atk_model, test_loader, cfg.device,
                                                                  epoch=step, debug=True)
            if topk_acc["top1"] > best_acc:
                best_acc = topk_acc["top1"]
            ckpt_path = osp.join(cfg.models_dir, f"{cfg.model2}/final_ckpt_s{cfg.seed}_t{step}.pth")
            torch.save(
                {
                    'top1_acc': topk_acc["top1"],
                    'top3_acc': topk_acc["top3"],
                    'top5_acc': topk_acc["top5"],
                    'iters': step,
                    'seed': cfg.seed,
                    'state_dict': atk_model.cpu().state_dict()
                },
                ckpt_path,
            )
            print(f"-> save: {cfg.model2} step:{step} seed:{cfg.seed}\n")


def main(args):
    helper.set_default_seed(args.seed)
    args.arch, args.dataset = args.model1.split("(")[1].split(")")[0].split(",")
    print(f"-> arch:{args.arch}, dataset:{args.dataset}")
    # load dataset
    train_loader = dloader.get_dataloader(dataset_id=args.subset, split="train",
                                         batch_size=args.batch_size, shuffle=True)
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test",
                                        batch_size=1000, shuffle=False)
    # load models
    benchmk = ImageBenchmark(archs=[args.arch], datasets=[args.dataset],
        datasets_dir=args.datasets_dir, models_dir=args.models_dir)
    target_model = benchmk.load_wrapper(args.model1, seed=1000).load_torch_model()
    surrogate_model = mloader.load_model(dataset_id=args.dataset, arch_id=args.arch).to(args.device)

    # load config
    cfg = dloader.load_cfg(dataset_id=args.dataset)
    helper.set_default_seed(args.seed)
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.backends = args.backends
    cfg.nb_stolen = args.nb_stolen
    cfg.batch_size = args.batch_size
    cfg.models_dir = args.models_dir
    cfg.batch_size_query = args.batch_size_query
    cfg.query_budget = int(args.nb_stolen * 20010)
    cfg.model2 = f"{args.model1}knockoff({args.arch},{args.subset})-"
    ckpt_root = osp.join(args.models_dir, cfg.model2)
    cfg.lr = 0.1 * (0.8 + 0.5 * random.random())
    if not osp.exists(ckpt_root):
        os.makedirs(ckpt_root)
    train_knockoff_random_strategy(cfg, vic_model=target_model, atk_model=surrogate_model, train_loader=train_loader, test_loader=test_loader)



if __name__ == "__main__":
    main(get_args())














