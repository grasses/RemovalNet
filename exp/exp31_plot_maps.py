import os.path as osp
import torch
import numpy as np
import argparse
from benchmark import ImageBenchmark
from utils import metric, ops, helper, vis
from dataset import loader as dloader

ops.set_default_seed(100)


def get_args():
    parser = argparse.ArgumentParser(description="Build micro benchmark.")
    parser.add_argument("-datasets_dir", action="store", dest="datasets_dir", default=osp.join(helper.ROOT, "dataset/data"),
                        help="Path to the dir of datasets.")
    parser.add_argument("-models_dir", action="store", dest="models_dir", default=osp.join(helper.ROOT, "model/ckpt"),
                        help="Path to the dir of benchmark models.")
    parser.add_argument("-model1", type=str, required=True, help="model_T")
    parser.add_argument("-model2", type=str, required=True, help="model_S")
    parser.add_argument("-batch_size", default=100, type=int, help="GPU device id")
    parser.add_argument("-t", type=int, required=True, help="model iteration")
    parser.add_argument("-seed", default=100, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    args, unknown = parser.parse_known_args()

    args.layers = ["layer1", "layer2", "layer3", "layer4", "layer5"]
    if "densenet" in args.model1:
        args.layers = ["features.pool0", "features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"]
    elif "mobile" in args.model1:
        args.layers = ["features.3", "features.6", "features.10", "features.16", "features.18"]
    elif "resnet" in args.model1:
        args.layers = ["layer1", "layer2", "layer3", "layer4", "layer3"]
    return args


args = get_args()
ops.set_default_seed(args.seed)


def main():
    arch, dataset = args.model1.split("(")[1].split(")")[0].split(",")
    print(args.model1, arch, dataset, args.seed)

    test_loader = dloader.get_dataloader(dataset_id=dataset, split="test", shuffle=False, batch_size=args.batch_size)
    benchmk = ImageBenchmark(datasets_dir=args.datasets_dir, models_dir=args.models_dir, archs=[arch], datasets=dataset)
    model_T = benchmk.load_wrapper(args.model1, seed=args.seed).load_torch_model()
    model_S = benchmk.load_wrapper(args.model2, seed=args.seed).load_torch_model()
    ckpt = osp.join(args.models_dir, model_S.task, f'final_ckpt_s{args.seed}_t{args.t}.pth')
    weights = torch.load(ckpt, map_location="cpu")["state_dict"]
    model_S.load_state_dict(weights)

    model_T.to(args.device)
    model_S.to(args.device)

    def get_eval_batch(model, loader, device, preview_off=30, preview_size=30):
        eval_x, eval_y = next(iter(loader))
        eval_x = eval_x[preview_off:preview_off + preview_size]
        mean, std = test_loader.mean, test_loader.std
        eval_ori_x = test_loader.unnormalize(eval_x, mean, std, clamp=True)[:preview_size]
        with torch.no_grad():
            eval_y = model(eval_x.to(device)).argmax(dim=1).detach().cpu()
            return [eval_x, eval_y, eval_ori_x]
    eval_x, eval_y, eval_ori_x = get_eval_batch(model=model_T, loader=test_loader, device=args.device)

    for l in np.arange(2, 4):
        fig_path = osp.join("output/exp", f"LayerCam_{args.layers[l]}_t{args.t}")
        vis.view_layer_activation(model_T, model_S, x=eval_x.clone(), y=eval_y.clone(),
                                  ori_x=eval_ori_x.clone(),
                                  size=len(eval_x), target_layer=args.layers[l],
                                  fig_path=fig_path, device=args.device)


if __name__ == "__main__":
    main()




















