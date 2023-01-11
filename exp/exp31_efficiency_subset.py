#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/01/10, homeway'


import copy
import os, argparse
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from defense.ZEST.zest import ZEST
from defense.DeepJudge import DeepJudge
from defense.ModelDiff import ModelDiff
from defense.IPGuard import IPGuard
from utils import helper, metric, vis
from . import ops
from benchmark import ImageBenchmark
from dataset import loader as dloader


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-tag", required=False, type=str, help="tag of script.")
    parser.add_argument("-dataset", choices=["CIFAR10", "CINIC10", "CelebA32+20", "CelebA32+31", "ImageNet"], help="Dataset")
    parser.add_argument("-batch_size", required=False, type=int, default=1000, help="tag of script.")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-start", default=100, type=int, help="Gap between two pretrained model")
    parser.add_argument("-gap", default=100, type=int, help="Gap between two pretrained model")
    parser.add_argument("-djm", required=False, type=float, default=3, help="m of DeepJudge")
    parser.add_argument("-layer_index", action="store", default=2, type=int, choices=[1, 2, 3, 4, 5], help="Layer Index")
    parser.add_argument("-seed_method", action="store", default="PGD", type=str, choices=["FGSM", "PGD", "CW"],
                        help="Type of blackbox generation")
    parser.add_argument("-test_size", required=False, type=int, default=400, help="test size of DeepJudge")
    parser.add_argument("-epsilon", action="store", default=0.2, type=float, help="Epsilon of ModelDiff")
    parser.add_argument("-k", action="store", default=0.01, type=float, help="k of IPGuard")
    parser.add_argument("-targeted", action="store", default="L", type=str, help="L:lest-likely R:random",
                        choices=["L", "R"])
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
    helper.set_default_seed(args.seed)
    args.djm = round(float(args.djm), 1)
    return args



def eval_DeepJudge(args, model1, model2, dir_list, test_loader):
    metrics = ["LOD", "LAD"]
    methods = ["removalnet", "distill", "finetune", "prune", "negative"]
    tag = f"{args.dataset}_{args.arch}_L{args.layer_index}_m{args.djm}"
    dj_exp_path = osp.join(args.out_root, f"DeepJudge/exp/exp11_{tag}.pt")
    dj_results = torch.load(dj_exp_path)

    results = {"LOD": [], "LAD": []}
    deepjudge = DeepJudge(model1, model2, test_loader=test_loader,
                          device=args.device, seed=args.seed, out_root=osp.join(args.out_root, "DeepJudge"),
                          batch_size=args.batch_size, test_size=args.test_size,
                          layer_index=args.layer_index, m=args.djm,
                          seed_method=args.seed_method)
    fingerprint = deepjudge.extract()
    phar = tqdm(enumerate(dir_list))
    for idx, dir_name in phar:
        removal_name = f"train({args.arch}, {args.dataset})-_{dir_name}"
        dj_removalnet_results = {removal_name: {"LOD": [], "LAD": []}}
        for step in np.arange(820, 1001, 20):
            ckpt = osp.join(args.models_dir, dir_name, f"final_ckpt_s{args.seed}_t{int(step)}.pth")
            model2.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            deepjudge.model2 = model2
            dist = deepjudge.verify(fingerprint)
            dj_removalnet_results[removal_name]["LOD"].append(dist["LOD"])
            dj_removalnet_results[removal_name]["LAD"].append(dist["LAD"])
            phar.set_description(f"-> model:{dir_name.split('-')[1]} seed:{step} dist:{dist}")
        dj_removalnet_results.update(dj_results)
        data = ops.exp11_normalize(dj_removalnet_results, methods=methods, metrics=metrics, defense_method="DeepJudge")
        results["LOD"].append(data["dists_nz"][0][0].tolist())
        results["LAD"].append(data["dists_nz"][1][0].tolist())
        print(f"-> idx:{idx}: results:{results}")
    return results


def eval_ZEST(args, model1, model2, dir_list, test_loader):
    metrics = ["L2", "cosine"]
    methods = ["removalnet", "distill", "finetune", "prune", "negative"]
    tag = f"{args.dataset}_{args.arch}"
    zest_path = osp.join(args.out_root, f"ZEST/exp/exp11_{tag}.pt")
    zest_results = torch.load(zest_path)

    results = {"L2": [], "cosine": []}
    phar = tqdm(enumerate(dir_list))
    for idx, dir_name in phar:
        removal_name = f"train({args.arch}, {args.dataset})-_{dir_name}"
        zest_removalnet_results = {removal_name: {"L2": [], "cosine": []}}
        for step in np.arange(820, 1001, 20):
            ckpt = osp.join(args.models_dir, dir_name, f"final_ckpt_s{args.seed}_t{int(step)}.pth")
            model2.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            zest = ZEST(model1, model2, test_loader=test_loader, device=args.device, out_root=osp.join(args.out_root, "ZEST"), seed=model2.seed)
            fingerprint = zest.extract(cache=False)
            dist = zest.verify(fingerprint)
            zest_removalnet_results[removal_name]["L2"].append(dist["L2"])
            zest_removalnet_results[removal_name]["cosine"].append(dist["cosine"])
            phar.set_description(f"-> model:{dir_name.split('-')[1]} seed:{step} dist:{dist}")
        zest_removalnet_results.update(zest_results)
        data = ops.exp11_normalize(zest_removalnet_results, methods=methods, metrics=metrics, defense_method="ZEST")
        results["L2"].append(data["dists_nz"][0][0].tolist())
        results["cosine"].append(data["dists_nz"][1][0].tolist())
        print(f"-> idx:{idx}: results:{results}")
    return results


def eval_ModelDiff(args, model1, model2, dir_list, test_loader):
    metrics = ["DDM"]
    methods = ["removalnet", "distill", "finetune", "prune", "negative"]
    tag = f"{args.dataset}_{args.arch}_eps{args.epsilon}"
    mdiff_path = osp.join(args.out_root, f"ModelDiff/exp/exp11_{tag}.pt")
    mdiff_results = torch.load(mdiff_path)

    results = {"DDV": []}
    phar = tqdm(enumerate(dir_list))
    for idx, dir_name in phar:
        removal_name = f"train({args.arch}, {args.dataset})-_{dir_name}"
        mdiff_removalnet_results = {removal_name: {"DDM": []}}
        benchmk = ImageBenchmark(datasets_dir=args.datasets_dir,
                                 models_dir=args.models_dir,
                                 archs=[args.arch],
                                 datasets=[args.dataset])
        model2 = benchmk.load_wrapper(dir_name, seed=args.seed).load_torch_model()
        for step in np.arange(820, 1001, 20):
            modeldiff = ModelDiff(model1, model2, test_loader=test_loader, device=args.device,
                                  out_root=osp.join(args.out_root, "ModelDiff"), seed=args.seed,
                                  epsilon=float(args.epsilon))
            fingerprint = modeldiff.extract(cache=False)
            dist = modeldiff.verify(fingerprint)
            mdiff_removalnet_results[removal_name]["DDM"].append(dist["DDM"])
            phar.set_description(f"-> model:{dir_name.split('-')[1]} seed:{step} dist:{dist}")
        mdiff_removalnet_results.update(mdiff_results)
        data = ops.exp11_normalize(mdiff_removalnet_results, methods=methods, metrics=metrics, defense_method="ModelDiff")
        results["DDV"].append(data["dists_nz"][0][0].tolist())
        print(f"-> idx:{idx}: results:{results}")
    return results


def eval_IPGuard(args, model1, model2, dir_list, test_loader):
    metrics = ["MR"]
    methods = ["removalnet", "distill", "finetune", "prune", "negative"]
    tag = f"{args.dataset}_{args.arch}_t{args.targeted}k{args.k}"
    ipguard_path = osp.join(args.out_root, f"IPGuard/exp/exp11_{tag}.pt")
    ipguard_results = torch.load(ipguard_path)

    results = {"MR": []}
    phar = tqdm(enumerate(dir_list))
    for idx, dir_name in phar:
        removal_name = f"train({args.arch}, {args.dataset})-_{dir_name}"
        ipguard_removalnet_results = {removal_name: {"MR": []}}
        benchmk = ImageBenchmark(datasets_dir=args.datasets_dir,
                                 models_dir=args.models_dir,
                                 archs=[args.arch],
                                 datasets=[args.dataset])
        model2 = benchmk.load_wrapper(dir_name, seed=args.seed).load_torch_model()
        for step in np.arange(820, 1001, 20):
            ipguard = IPGuard(model1, model2, test_loader=test_loader, device=args.device,
                              out_root=osp.join(args.out_root, "IPGuard"),
                              k=args.k, targeted=args.targeted, test_size=args.test_size, seed=args.seed)
            fingerprint = ipguard.extract()
            dist = ipguard.verify(fingerprint)
            ipguard_removalnet_results[removal_name]["MR"].append(dist["MR"])
            phar.set_description(f"-> model:{dir_name.split('-')[1]} seed:{step} dist:{dist}")
        ipguard_removalnet_results.update(ipguard_results)
        data = ops.exp11_normalize(ipguard_removalnet_results, methods=methods, metrics=metrics,
                                   defense_method="IPGuard")
        results["MR"].append(data["dists_nz"][0][0].tolist())
        print(f"-> idx:{idx}: results:{results}")
    return results


def eval_accuracy(args, model, model_list, test_loader, seeds=True):
    results_acc = []
    for idx, dir_name in enumerate(model_list):
        results_acc.append([])
        if seeds:
            for step in np.arange(820, 1001, 20):
                ckpt = osp.join(args.models_dir, dir_name, f"final_ckpt_s{args.seed}_t{int(step)}.pth")
                model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
                _, topk, _ = metric.topk_test(model, test_loader=test_loader, device=args.device, epoch=step, debug=False)
                results_acc[idx].append(topk["top1"])
                print(f"-> model:{dir_name} seed:{step} accuracy:{topk['top1']}%")
            print()
        else:
            ckpt = osp.join(args.models_dir, dir_name, f"final_ckpt_s100_t1000.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            _, topk, _ = metric.topk_test(model, test_loader=test_loader, device=args.device, epoch=0, debug=False)
            results_acc[idx].append(topk["top1"])
            print(f"-> model:{dir_name} accuracy:{topk['top1']}%")
    return np.array(results_acc, dtype=np.float32)



def main(args):
    model_list1 = [
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.02,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.04,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.06,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.08,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.1,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.12,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.14,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.16,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.18,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(HAM10000,0.2,0.2,2.0,20,l2)-",
    ]
    model_list2 = [
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.02,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.04,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.06,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.08,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.1,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.12,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.14,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.16,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.18,0.2,2.0,20,l2)-",
        "train(vgg19_bn,HAM10000)-removalnet(BCN20000,0.2,0.2,2.0,20,l2)-",
    ]
    target_model = model_list1[0].split("-")[0] + "-"
    args.arch, args.dataset = model_list1[0].split("-")[0].split("(")[1].split(")")[0].split(",")
    subset1 = model_list1[0].split("-")[1].split("(")[1].split(")")[0].split(",")[0]
    subset2 = model_list2[0].split("-")[1].split("(")[1].split(")")[0].split(",")[0]

    results = {
        "config": {
            "subset1": subset1,
            "subset2": subset2,
            "model_list1": model_list1,
            "model_list2": model_list2,
            "model_name1": f"Surrogate-{subset1}",
            "model_name2": f"Surrogate-{subset2}",
            "xticks": [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        },
        "plot_acc": {
            "target": None,
            "model1": None,
            "model2": None,
        },
        "plot_distance": {
            "model1": {},
            "model2": {}
        },
        "plot_similarity": {
            "model1": {},
            "model2": {}
        }
    }
    exp_path = osp.join(args.out_root, "exp", f"exp31_efficiency_subset_{args.arch}_{args.dataset}.pt")
    if osp.exists(exp_path):
        results = torch.load(exp_path)
    config = results["config"]

    # step1: eval accuracy
    benchmk = ImageBenchmark(datasets_dir=args.datasets_dir,
                             models_dir=args.models_dir,
                             archs=[args.arch],
                             datasets=[args.dataset])
    model = benchmk.load_wrapper(model_list1[0], seed=args.seed).load_torch_model()
    test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", shuffle=True, batch_size=args.batch_size)
    target_model = benchmk.load_wrapper(target_model, seed=1000).load_torch_model()

    if results["plot_acc"]["target"] is None:
        _, topk, _ = metric.topk_test(target_model, test_loader=test_loader, device=args.device, debug=True)
        results["plot_acc"]["target"] = topk["top1"]
        results["plot_acc"]["model1"] = eval_accuracy(args, model, model_list1, test_loader=test_loader)
        results["plot_acc"]["model2"] = eval_accuracy(args, model, model_list2, test_loader=test_loader)
    acc1 = results["plot_acc"]["model1"]
    acc2 = results["plot_acc"]["model2"]
    target_acc = results["plot_acc"]["target"]
    legends = ["Target", config["model_name1"], config["model_name2"]]
    pdf_path = osp.join(args.out_root, "pdf", f"exp31_{args.arch}_{args.dataset}_accuracy.pdf")
    vis.plot_exp31_subset_variance_accuracy(target_acc, acc1, acc2, xticks=results["config"]["xticks"],
                                            path=pdf_path, legends=legends)


    # step2: eval model distance & similarity
    # step2.1: eval DeepJudge
    if "DeepJudge-LOD" not in results["plot_distance"]["model1"].keys():
        res = eval_DeepJudge(args, target_model, model2=model, dir_list=model_list1, test_loader=test_loader)
        results["plot_distance"]["model1"]["DeepJudge-LOD"] = np.array(res["LOD"], dtype=np.float32)
        results["plot_distance"]["model1"]["DeepJudge-LAD"] = np.array(res["LAD"], dtype=np.float32)
        res = eval_DeepJudge(args, target_model, model2=model, dir_list=model_list1, test_loader=test_loader)
        results["plot_distance"]["model2"]["DeepJudge-LOD"] = np.array(res["LOD"], dtype=np.float32)
        results["plot_distance"]["model2"]["DeepJudge-LAD"] = np.array(res["LAD"], dtype=np.float32)
        if osp.exists(exp_path):
            cache = torch.load(exp_path)
            cache["plot_distance"]["model1"].update(results["plot_distance"]["model1"])
            cache["plot_distance"]["model2"].update(results["plot_distance"]["model2"])
            torch.save(cache, exp_path)
        else:
            torch.save(results, exp_path)

    # step2.2: eval ZEST
    if "ZEST-L2" not in results["plot_distance"]["model2"].keys():
        test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", shuffle=True, batch_size=128)
        res1 = eval_ZEST(args, target_model, model2=model, dir_list=model_list1, test_loader=test_loader)
        results["plot_distance"]["model1"]["ZEST-L2"] = np.array(res1["L2"], dtype=np.float32)
        results["plot_distance"]["model1"]["ZEST-cosine"] = np.array(res1["cosine"], dtype=np.float32)

        res2 = eval_ZEST(args, target_model, model2=model, dir_list=model_list2, test_loader=test_loader)
        results["plot_distance"]["model2"]["ZEST-L2"] = np.array(res2["L2"], dtype=np.float32)
        results["plot_distance"]["model2"]["ZEST-cosine"] = np.array(res2["cosine"], dtype=np.float32)
        if osp.exists(exp_path):
            cache = torch.load(exp_path)
            cache["plot_distance"]["model1"].update(results["plot_distance"]["model1"])
            cache["plot_distance"]["model2"].update(results["plot_distance"]["model2"])
            torch.save(cache, exp_path)
        else:
            torch.save(results, exp_path)

    results = torch.load(exp_path)
    model1_results = results["plot_distance"]["model1"]
    model2_results = results["plot_distance"]["model2"]
    pdf_path = osp.join(args.out_root, "pdf", f"exp31_{args.arch}_{args.dataset}_distance.pdf")
    vis.plot_exp31_subset_variance_distance(model1_results, model2_results, xticks=results["config"]["xticks"],
                                            path=pdf_path, legends=[config["model_name1"], config["model_name2"]])


    # step2.3: ModelDiff
    if "ModelDiff-DDV" not in results["plot_similarity"]["model1"].keys():
        test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", shuffle=True, batch_size=500)
        res1 = eval_ModelDiff(args, target_model, model2=model, dir_list=model_list1, test_loader=test_loader)
        results["plot_similarity"]["model1"]["ModelDiff-DDV"] = np.array(res1["DDV"], dtype=np.float32)

        res2 = eval_ModelDiff(args, target_model, model2=model, dir_list=model_list2, test_loader=test_loader)
        results["plot_similarity"]["model2"]["ModelDiff-DDV"] = np.array(res2["DDV"], dtype=np.float32)
        if osp.exists(exp_path):
            cache = torch.load(exp_path)
            cache["plot_similarity"]["model1"].update(results["plot_similarity"]["model1"])
            cache["plot_similarity"]["model2"].update(results["plot_similarity"]["model2"])
            torch.save(cache, exp_path)
        else:
            torch.save(results, exp_path)


    # step2.4: IPGuard
    if "IPGuard-MR" not in results["plot_similarity"]["model2"].keys():
        test_loader = dloader.get_dataloader(dataset_id=args.dataset, split="test", shuffle=True, batch_size=500)
        res1 = eval_IPGuard(args, target_model, model2=model, dir_list=model_list1, test_loader=test_loader)
        results["plot_similarity"]["model1"]["IPGuard-MR"] = np.array(res1["MR"], dtype=np.float32)

        res2 = eval_IPGuard(args, target_model, model2=model, dir_list=model_list2, test_loader=test_loader)
        results["plot_similarity"]["model2"]["IPGuard-MR"] = np.array(res2["MR"], dtype=np.float32)
        if osp.exists(exp_path):
            cache = torch.load(exp_path)
            cache["plot_similarity"]["model1"].update(results["plot_similarity"]["model1"])
            cache["plot_similarity"]["model2"].update(results["plot_similarity"]["model2"])
            torch.save(cache, exp_path)
        else:
            torch.save(results, exp_path)

    results = torch.load(exp_path)
    model1_results = results["plot_similarity"]["model1"]
    model2_results = results["plot_similarity"]["model2"]
    pdf_path = osp.join(args.out_root, "pdf", f"exp31_{args.arch}_{args.dataset}_similarity.pdf")
    vis.plot_exp31_subset_variance_distance(model1_results, model2_results, xticks=results["config"]["xticks"],
                                            path=pdf_path,
                                            legends=[config["model_name1"], config["model_name2"]])


if __name__ == "__main__":
    args = get_args()
    main(args)


















