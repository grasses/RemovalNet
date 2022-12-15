import os.path as osp
import torch
import numpy as np
import argparse
from benchmark import ImageBenchmark
from utils import metric, ops, helper, vis
from dataset import loader as dloader
from torch.nn import functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
    parser.add_argument("-test_size", default=100, type=int, help="GPU device id")
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



def scatter_points(data_dict, label, file_path, num_classes, lims=100, fontsize=28):
    plt.figure(figsize=(16, 16), dpi=160)
    plt.cla()
    keys = list(data_dict.keys())

    d1 = data_dict[keys[0]]
    d2 = data_dict[keys[1]]
    d = np.concatenate([d1, d2])
    xx, yy = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))
    knn = KNeighborsClassifier(n_neighbors=num_classes).fit(d, label)
    zz = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = np.reshape(zz, xx.shape)
    plt.pcolormesh(xx, yy, zz, cmap=plt.get_cmap('Greys'), vmin=0, vmax=num_classes-1, alpha=0.1)
    for idx, (key, value) in enumerate(data_dict.items()):
        plt.scatter(value[:, 0], value[:, 1], lw=6, s=60, label=key, marker=vis.markers[idx], alpha=0.7)

    xy = data_dict["model_T"]
    for l in range(num_classes):
        off = int(l * args.test_size)
        x = float(np.mean(xy[off: off + args.test_size, 0])) + 1
        y = float(np.mean(xy[off: off + args.test_size, 1]))
        plt.text(x, y, f"y={l}", fontdict = {
            'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 35,
        })
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right", numpoints=2, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


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
    model_T.eval()
    model_S.eval()

    logit_T, xs, cs = {}, {}, {}
    label_TS = []
    for l in range(test_loader.num_classes):
        xs[l] = []
        cs[l] = []
        logit_T[l] = []
        label_TS.append(np.ones([args.test_size]) * l)
    label_TS = np.array(np.concatenate(label_TS), dtype=np.int32)
    label_TS = np.concatenate([label_TS, label_TS])

    # step1: select 100 sample for each class
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            pred_l = model_T(x).detach().cpu()
            pred_c = F.softmax(pred_l, dim=1)
            pred_y = pred_c.argmax(dim=1)
            x = x.cpu()
            for l in range(test_loader.num_classes):
                idx = (pred_y == l).nonzero(as_tuple=True)[0]
                xs[l].append(x[idx])
                cs[l].append(pred_c[idx, l])
                logit_T[l].append(pred_l[idx])
            if step > 200:
                break

    # step2: find confidence first 100
    for l in range(test_loader.num_classes):
        xs[l] = torch.cat(xs[l])
        cs[l] = torch.cat(cs[l])
        logit_T[l] = torch.cat(logit_T[l])
        idx = torch.argsort(cs[l], descending=True)
        xs[l] = xs[l][idx[:args.test_size]]
        cs[l] = cs[l][idx[:args.test_size]]
        logit_T[l] = logit_T[l][idx[:args.test_size]]

    # step3: embedding the logits
    logit = []
    logit.append(torch.cat(list(logit_T.values())))
    for idx, t in enumerate([100, 500, 1000]):
        ckpt = osp.join(args.models_dir, model_S.task, f'final_ckpt_s{args.seed}_t{t}.pth')
        print(f"-> load model:{ckpt}")
        weights = torch.load(ckpt, map_location="cpu")["state_dict"]
        model_S.load_state_dict(weights)
        model_S.to(args.device)
        model_S.eval()

        logit_S = []
        for l in range(test_loader.num_classes):
            #out = model_S(xs[l].to(args.device)).detach().cpu()
            out = ops.batch_forward(model_S, xs[l], batch_size=100).detach().cpu()
            logit_S.append(out)
        logit.append(torch.cat(logit_S))

    feats = F.softmax(torch.cat(logit), dim=1).detach().cpu().numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(feats)
    num_sample = int(test_loader.num_classes * args.test_size)
    for idx, t in enumerate([0, 100, 500, 1000]):
        off = int(idx * num_sample)
        data = {
            "model_T": xy[0: num_sample],
            f"model_S(t={t})": xy[off: off + num_sample],
        }
        fig_path = osp.join(helper.ROOT, f"output/exp/exp32_dcb_{args.model1}_t{t}.step")
        scatter_points(data, label=label_TS, file_path=f"{fig_path}_t{t}.pdf", num_classes=test_loader.num_classes, lims=100)



if __name__ == "__main__":
    main()




















