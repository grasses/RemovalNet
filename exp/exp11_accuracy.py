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
    parser.add_argument("-start", type=int, default=820, help="model iteration")
    parser.add_argument("-gap", default=20, type=int, help="model iteration")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()
    return args


args = get_args()
ops.set_default_seed(args.seed)


def main():
    arch, dataset = args.model.split("(")[1].split(")")[0].split(",")
    print(args.model, arch, dataset, args.seed)

    test_loader = dloader.get_dataloader(dataset_id=dataset, split="test", shuffle=False, batch_size=args.batch_size)
    benchmk = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir, archs=[arch], datasets=dataset)
    surrogate_model = benchmk.load_wrapper(args.model, seed=args.seed).load_torch_model()

    # eval accuracy of target model
    target_model = args.model.split("-")[0] + "-"
    target_model = benchmk.load_wrapper(target_model, seed=args.seed).load_torch_model()
    _, target_acc, _ = metric.topk_test(target_model, test_loader, device=args.device, epoch=0, debug=True)
    target_acc = target_acc["top1"]

    # eval accuracy of surrogate model
    surr_acc = []
    for t in np.arange(args.start, 1001, args.gap):
        ckpt = osp.join(args.models_dir, surrogate_model.task, f'final_ckpt_s{args.seed}_t{t}.pth')
        surrogate_model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        surrogate_model.to(args.device)
        _, surrogate_acc, _ = metric.topk_test(surrogate_model, test_loader, device=args.device, epoch=t, debug=True)
        surr_acc.append(surrogate_acc["top1"])
    surr_acc = np.array(surr_acc, dtype=np.float32)
    surr_acc_min = float(np.min(surr_acc))
    surr_acc_max = float(np.max(surr_acc))
    med = (surr_acc_max - surr_acc_min) / 2.0 + surr_acc_min
    print(f"-> accuarcy:{round(med, 2)}±{round(surr_acc_max - surr_acc_min, 2)} min:{round(surr_acc_min, 2)} max:{round(surr_acc_max, 2)}"
        f" drop:{round(target_acc - surr_acc_max, 2)}~{round(target_acc - surr_acc_min, 2)} drop_mean:{round(target_acc-np.mean(surr_acc), 2)}")


if __name__ == "__main__":
    main()
