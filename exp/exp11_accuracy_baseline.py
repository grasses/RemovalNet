import os.path as osp
import torch
import numpy as np
import argparse
from benchmark import ImageBenchmark
from utils import metric, ops, helper
from dataset import loader as dloader


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model", type=str, required=True, help="model")
    parser.add_argument("-batch_size", default=100, type=int, help="GPU device id")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()
    return args


args = get_args()
ops.set_default_seed(args.seed)


def main():
    args = get_args()
    print(f"-> Running with config:{args}")

    arch, dataset = args.model.split("(")[1].split(")")[0].split(",")
    print(args.model, arch, dataset, args.seed)

    test_loader = dloader.get_dataloader(dataset_id=dataset, split="test", shuffle=False, batch_size=args.batch_size)
    benchmk = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir, archs=[arch], datasets=[dataset])

    # eval accuracy of target model
    target_model = args.model.split("-")[0] + "-"
    target_model = benchmk.load_wrapper(target_model, seed=args.seed).load_torch_model()
    _, target_acc, _ = metric.topk_test(target_model, test_loader, device=args.device, epoch=0, debug=True)
    target_acc = target_acc["top1"]

    # eval baseline
    cfg = dloader.load_cfg(dataset_id=dataset, arch_id=[arch])
    benchmk = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir, archs=[arch], datasets=[dataset])
    methods = ["distill", "finetune", "prune", "negative", "steal"]
    models = benchmk.list_models(cfg=cfg, methods=methods)

    done_list = []
    for idx, model in enumerate(models):
        model = model.load_torch_model(seed=1000)
        if idx == 0: continue
        if str(model.task) in done_list: continue

        surr_acc = []
        for seed in np.arange(100, 1001, 100):
            ckpt = osp.join(args.models_dir, model.task, f'final_ckpt_s{seed}.pth')
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
            _, acc, _ = metric.topk_test(model, test_loader, device=args.device, epoch=seed, debug=True)
            surr_acc.append(acc["top1"])
        surr_acc = np.array(surr_acc, dtype=np.float32)
        surr_acc_min = float(np.min(surr_acc))
        surr_acc_max = float(np.max(surr_acc))
        med = (surr_acc_max - surr_acc_min) / 2.0 + surr_acc_min
        print(
            f"-> model:{model.task} accuarcy:{round(med, 2)}±{round(surr_acc_max - surr_acc_min, 2)} min:{round(surr_acc_min, 2)} max:{round(surr_acc_max, 2)}"
            f" drop:{round(acc['top1'] - surr_acc_max, 2)}~{round(acc['top1'] - surr_acc_min, 2)} drop_mean:{round(target_acc - np.mean(surr_acc), 2)}")
        done_list.append(str(model.task))

if __name__ == "__main__":
    main()









