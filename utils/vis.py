#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import copy
import torch
import numpy as np
import os.path as osp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import helper
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
sys_args = helper.get_args()
colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
patterns = ['/', '\\', 'xx', 'x', '\\\\', '//', '+', '..', '++']
markers = ["o", "v", "s", "^", "d", "*", "D", "p", "+", "X"]
fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 30,
}

def plot_exp41_impact_alpha_beta(result_distance, result_similarity, group, xticks, path, fontsize=30, linewidth=6, markersize=20):
    plt.figure(figsize=(16, 12), dpi=200)
    plt.cla()
    plt.grid()

    cnt = 0
    for idx, metric in enumerate(result_distance.keys()):
        item = np.array(result_distance[metric][group])
        plt.bar(np.arange(len(item)) + 1.0 + cnt*0.22, item, width=0.2, color=colors[cnt], label=metric, hatch=patterns[cnt])
        cnt += 1
        xticks = np.arange(1, len(item) + 1)
    yticks = np.arange(0, 1.01, 0.2).tolist()
    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.xlabel("alpha/beta", fontsize=fontsize - 5)
    plt.ylabel("Distance (normalize)", fontsize=fontsize)
    plt.legend(loc="best", numpoints=1, fontsize=fontsize - 5)
    plt.savefig(path.split(".pdf")[0] + "-distance.pdf")


    plt.figure(figsize=(16, 12), dpi=200)
    plt.cla()
    plt.grid()
    cnt = 0
    for idx, metric in enumerate(result_similarity.keys()):
        item = np.array(result_similarity[metric][group])
        plt.bar(np.arange(len(item)) + 1.0 + cnt * 0.22, item, width=0.2, color=colors[cnt], label=metric, hatch=patterns[cnt])
        cnt += 1
        xticks = np.arange(1, len(item)+1)

    yticks = np.arange(0, 1.01, 0.2).tolist()
    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.xlabel("alpha/beta", fontsize=fontsize - 5)
    plt.ylabel("Similarity (normalize)", fontsize=fontsize)
    plt.legend(loc="best", numpoints=1, fontsize=fontsize - 5)
    plt.savefig(path.split(".pdf")[0] + "-similarity.pdf")
    print(f"-> saving fig: {path}")


def plot_exp32_computation_variance_accuracy(removal1, removal2, knockoff1, knockoff2, legends, xticks, path, fontsize=30, linewidth=6, markersize=20):
    #plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(16, 12), dpi=200)
    plt.cla()
    plt.grid()
    yticks = np.arange(0, 101, 20).tolist()

    removal1_mean = removal1.mean()
    removal2_mean = removal2.mean()
    #plt.text(xticks[1], removal1_mean + 8, f"{legends[0]} :{round(float(removal1_mean), 2)}%", fontdict=fontdict, color="black")
    #plt.text(xticks[3], 70, f"{legends[1]} :{round(float(removal2_mean), 2)}%", fontdict=fontdict, color="black")

    print("-> legends1", legends)
    for idx, l in enumerate(legends):
        legends[idx] = l.replace("GTSRB+1", "GTSRB-Sub1").replace("GTSRB+2", "GTSRB-Sub2")
    print("-> legends2", legends)

    removal1_mean = 89.89
    removal2_mean = 87.15
    plt.hlines(removal1_mean, xmin=xticks[1], xmax=xticks[-1], linewidth=5, colors=colors[0], linestyles="dashed", label=legends[0])
    plt.hlines(removal2_mean, xmin=xticks[1], xmax=xticks[-1], linewidth=5, colors=colors[1], linestyles="dashed", label=legends[1])

    lower = [np.min(knockoff1[idx]) for idx in range(knockoff1.shape[0])]
    upper = [np.max(knockoff1[idx]) for idx in range(knockoff1.shape[0])]
    mean = [np.mean(knockoff1[idx]) for idx in range(knockoff1.shape[0])]
    plt.fill_between(xticks, lower, upper, color=colors[2], alpha=0.5)
    plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[2], c=colors[2], label=legends[2])

    lower = [np.min(knockoff2[idx]) for idx in range(knockoff2.shape[0])]
    upper = [np.max(knockoff2[idx]) for idx in range(knockoff2.shape[0])]
    mean = [np.mean(knockoff2[idx]) for idx in range(knockoff2.shape[0])]
    plt.fill_between(xticks, lower, upper, color=colors[3], alpha=0.5)
    plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[3], c=colors[3], label=legends[3])

    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.xlabel("Iterations", fontsize=fontsize - 5)
    plt.ylabel("Accuracy (%)", fontsize=fontsize)
    plt.legend(loc="lower right", numpoints=1, fontsize=fontsize)
    plt.savefig(path)
    print(f"-> saving fig: {path}")


def plot_exp31_subset_variance_accuracy(target_acc, model1_acc, model2_acc, legends, xticks, path, fontsize=35, linewidth=5, markersize=25):
    plt.figure(figsize=(18, 14), dpi=200)
    plt.cla()
    plt.grid()

    #plt.text(0.03, target_acc - 5, f"Target Model:{round(float(target_acc), 2)}%", fontdict=fontdict, color="black")
    plt.hlines(target_acc, xmin=xticks[0], xmax=xticks[-1], linewidth=linewidth, colors="gray", linestyles="dashed", label="Target Model")

    lower = [np.min(model1_acc[idx]) for idx in range(model1_acc.shape[0])]
    upper = [np.max(model1_acc[idx]) for idx in range(model1_acc.shape[0])]
    mean = [np.mean(model1_acc[idx]) for idx in range(model1_acc.shape[0])]
    plt.fill_between(xticks, lower, upper, color=colors[0], alpha=0.2)
    plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[0], c=colors[0], label=legends[1])

    lower = [np.min(model2_acc[idx]) for idx in range(model2_acc.shape[0])]
    upper = [np.max(model2_acc[idx]) for idx in range(model2_acc.shape[0])]
    mean = [np.mean(model2_acc[idx]) for idx in range(model2_acc.shape[0])]
    plt.fill_between(xticks, lower, upper, color=colors[1], alpha=0.2)
    plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[1], c=colors[1], label=legends[2])

    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(np.arange(0, 101, 20).tolist(), fontsize=fontsize+10)
    plt.xlabel("Data holding rate", fontsize=fontsize)
    plt.ylabel("Accuracy (%)", fontsize=fontsize)
    plt.legend(loc="lower right", numpoints=1, fontsize=fontsize)
    plt.savefig(path)
    print(f"-> saving fig: {path}")


def plot_exp31_subset_variance_distance(model1_results, model2_results, legends, xticks, path, fontsize=35, linewidth=5, markersize=25):
    plt.figure(figsize=(18, 14), dpi=200)
    plt.cla()
    plt.grid()

    cnt = 0
    distance_flag = False
    metrics = list(model1_results.keys())
    for idx, metric in enumerate(metrics):
        lower = [np.min(model1_results[metric][idx]) for idx in range(len(xticks))]
        upper = [np.max(model1_results[metric][idx]) for idx in range(len(xticks))]
        mean = [np.mean(model1_results[metric][idx]) for idx in range(len(xticks))]
        plt.fill_between(xticks, lower, upper, color=colors[cnt], alpha=0.2)
        plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[cnt],
                 c=colors[cnt], label=f"{legends[0]}: {metric}")
        cnt += 1
        if "DeepJudge" in metric:
            distance_flag = True

    for idx, metric in enumerate(metrics):
        lower = [np.min(model2_results[metric][idx]) for idx in range(len(xticks))]
        upper = [np.max(model2_results[metric][idx]) for idx in range(len(xticks))]
        mean = [np.mean(model2_results[metric][idx]) for idx in range(len(xticks))]
        plt.fill_between(xticks, lower, upper, color=colors[cnt], alpha=0.2)
        plt.plot(xticks, mean, linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[cnt],
                 c=colors[cnt], label=f"{legends[1]}: {metric}")
        cnt += 1

    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(np.arange(0, 1.01, 0.2).tolist(), fontsize=fontsize+10)
    plt.xlabel("Data holding rate", fontsize=fontsize)
    if distance_flag:
        plt.ylabel("Distance (normalized)", fontsize=fontsize)
        plt.legend(loc="lower right", numpoints=1, fontsize=fontsize - 10)
    else:
        plt.ylabel("Similarity (normalized)", fontsize=fontsize)
        plt.legend(loc="upper right", numpoints=1, fontsize=fontsize)
    plt.savefig(path)
    print(f"-> saving fig: {path}")





def plot_embedding(xy, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(12, 12), dpi=160)
    plt.cla()
    plt.grid()
    size = int(len(xy) / 2)
    labels = ["model_0", "model_t"]
    for idx in range(2):
        off = int(idx * size)
        plt.scatter(xy[off:off + size, 0], xy[off:off + size, 1], lw=4, s=60, label=labels[idx])

    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="upper right", numpoints=1, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


def scatter_points(data_dict, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()

    for idx, (key, value) in enumerate(data_dict.items()):
        plt.scatter(value[:, 0], value[:, 1], lw=4, s=60, label=key, marker=markers[idx])
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="upper right", numpoints=1, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


def pixel_plot(data, ax, fontsize=18, hide_labels=False):
    pc = ax.pcolormesh(data, vmin=0, vmax=1)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc


def plot_accuracy_dist_curve(atk_data, neg_data, steps, legends, path, fontsize=25, linewidth=8, markersize=38):
    plt.figure(figsize=(16, 12), dpi=160)
    plt.cla()
    plt.grid()
    global markers
    fontdict["size"] = fontsize

    # configure for datasets
    conf = {
        "CIFAR10": {
            "yticks": np.arange(40, 101, 10).tolist(),
            "acc_y_offset": 5,
        },
        "CelebA": {
            "yticks": np.arange(80, 101, 5).tolist(),
            "acc_y_offset": 1,
        },
        "HAM10000": {
            "yticks": np.arange(50, 101, 10).tolist(),
            "acc_y_offset": 1,
        },
        "ImageNet": {
            "yticks": np.arange(40, 81, 10).tolist(),
            "acc_y_offset": 1,
        }
    }
    conf = conf["CIFAR10"]

    markers_neg = "P"
    if "ZEST" in legends[0]:
        markers = markers[2:]
        markers_neg = "X"

    # plot RemovalNet curve
    for idx, (metric, data) in enumerate(atk_data.items()):
        plt.plot(data[:, 1], data[:, 0], linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[idx], c=colors[idx])

    # plot accuracy of target model
    fontdict["size"] = 50
    target_acc = round(float(data[0, 0]), 2)
    #plt.text(0.03, target_acc + conf["acc_y_offset"], f"Acc:98.84%", fontdict=fontdict, color="black")
    plt.hlines(target_acc, xmin=0, xmax=1.0, linewidth=5, colors="gray", linestyles="dashed")

    # scatter negative points
    for idx, (metric, data) in enumerate(neg_data.items()):
        legends += [f"{legends[idx]}(negative)"]
        plt.scatter(data[:, 1], data[:, 0], s=markersize**2, marker=markers_neg, c=colors[idx])

    #plt.legend(labels=legends, loc='best', fontsize=fontsize)
    plt.xticks(np.arange(0, 1.01, 0.2).tolist(), fontsize=fontsize)
    plt.yticks(conf["yticks"], fontsize=fontsize)
    plt.xlabel("Distance", fontsize=fontsize)
    plt.ylabel("Accuracy (%)", fontsize=fontsize)
    plt.savefig(path)
    print(f"-> saving fig: {path}")


def view_learning_state(data, file_path, fontsize=30):
    for key in data["keys"]:
        plt.figure(figsize=(16, 12), dpi=100)
        plt.cla()
        plt.grid()
        x = np.array(data["t"], dtype=np.int32)
        y = np.array(data[key])

        plt.plot(x, y, label=f"Surrogate Model", linewidth=5, marker="*", markersize=10, linestyle="solid")
        plt.plot(x, np.repeat(np.min(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(np.argmin(y) * 20.0), np.min(y) - 3, round(float(np.min(y)), 2), fontsize=fontsize-5)
        plt.plot(x, np.repeat(np.max(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(np.argmax(y) * 20.0), np.max(y) + 3, round(float(np.max(y)), 2), fontsize=fontsize-5)
        last_y = float(y[-1])
        plt.text(float(len(y)-1) * 20.0, last_y + 3, round(last_y, 2), fontsize=fontsize - 5)


        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(key.upper(), fontsize=fontsize)
        if key == "acc":
            plt.ylim(0, 100.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="best", numpoints=1, fontsize=fontsize)
        fpath = file_path + f"_{key.upper()}.pdf"
        plt.savefig(fpath)
        print(f"-> saving fig: {fpath}")


def view_layer_activation(model_T, model_t, x, y, ori_x, target_layer, fig_path, size=8, fontsize=30, device=torch.device("cuda:0")):
    x, y = x.to(device), y.to(device)
    class_idx = y.tolist()
    model_0 = model_T.to(device)
    model_t = model_t.to(device)

    # extract LayerCAM for model_T
    CAM0 = LayerCAM(model_0, target_layer)
    cams = CAM0(class_idx=class_idx, scores=model_0(x))
    fused_cam0 = CAM0.fuse_cams(cams).detach().cpu()
    CAM0.remove_hooks()
    # extract LayerCAM for model_t
    CAM1 = LayerCAM(model_t, target_layer)
    cams = CAM1(class_idx=class_idx, scores=model_t(x))
    fused_camt = CAM1.fuse_cams(cams).detach().cpu()
    CAM1.remove_hooks()

    fig = plt.figure(figsize=(10, 4 + size * 2), dpi=200)
    img_idx = 0
    for idx in range(size):
        # original image
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_image = to_pil_image(ori_x[idx])
        plt.imshow(pil_image, interpolation='nearest')

        # overlay of target model output
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_map_T = to_pil_image(fused_cam0[idx].squeeze(0), mode='F')
        pil_overlay_T = overlay_mask(pil_image, pil_map_T, alpha=0.5)
        plt.imshow(pil_overlay_T, interpolation='nearest')

        # overlay of surrogate model output
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_map_t = to_pil_image(fused_camt[idx].squeeze(0), mode='F')
        pil_overlay_t = overlay_mask(pil_image, pil_map_t, alpha=0.5)
        plt.imshow(pil_overlay_t)
    pth = f"{fig_path}_OverLay.pdf"
    plt.savefig(pth)
    print(f"-> Saving OverLay:{pth}")

    fig = plt.figure(constrained_layout=True, figsize=(8, 4 + size * 3), dpi=200)
    # Left collum
    subfigs = fig.subfigures(1, 2, wspace=0.1, hspace=0.1)
    axsLeft = subfigs[0].subplots(size, 1)
    subfigs[0].set_facecolor('#ffffff')
    for nn, ax in enumerate(axsLeft):
        fmap = fused_cam0[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[0].suptitle(f'Model_T {target_layer}', fontsize=fontsize-2)
    subfigs[0].colorbar(pc, shrink=0.8, ax=axsLeft, location='bottom')
    # Right collum
    axsRight = subfigs[1].subplots(size, 1)
    subfigs[1].set_facecolor('#ffffff')
    for nn, ax in enumerate(axsRight):
        fmap = fused_camt[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[1].suptitle(f'Model_S {target_layer}', fontsize=fontsize-2)
    subfigs[1].colorbar(pc, shrink=0.8, ax=axsRight, location='bottom')
    fig.suptitle(f'LayerCAM of {target_layer}', fontsize=fontsize)
    pth = f"{fig_path}_ActativeLay.pdf"
    plt.savefig(pth)
    print(f"-> Saving ActativeLay:{pth}")


def view_decision_boundary(model1, model2, test_loader, fig_path, step, device=sys_args.device):
    model1 = model1.to(device)
    model2 = model2.to(device)

    logist1, logist2 = [], []
    select_t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    select_logist1, select_logist2 = [], []
    correct1, correct2, cnt = 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out1 = model1(x).detach().cpu()
            out2 = model2(x).detach().cpu()
            if i == 0:
                if out1.shape[1] < 100:
                    select_t = [1, 2, 3, 4]

            for t in select_t:
                idxs = (y == t).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    select_logist1.append(out1[idxs])
                    select_logist2.append(out2[idxs])
                if i > 15:
                    break

            if i <= 5:
                logist1.append(out1)
                logist2.append(out2)
                correct1 += y.cpu().eq(out1.argmax(dim=1).view_as(y)).sum()
                correct2 += y.cpu().eq(out2.argmax(dim=1).view_as(y)).sum()
                cnt += len(x)

    acc1 = round(100.0 * float(correct1 / cnt), 3)
    acc2 = round(100.0 * float(correct2 / cnt), 3)

    logist_array = torch.cat(logist1 + logist2).numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(logist_array)
    data = {
        "model_T": xy[:int(len(xy) / 2)],
        "model_S": xy[int(len(xy) / 2):],
    }
    scatter_points(data, file_path=f"{fig_path}_t{step}.pdf", lims=100)


    select_logist_array = torch.cat(select_logist1 + select_logist2).detach().cpu().numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(select_logist_array)
    name = ""
    for k in select_t:
        name += f"{k}-"
    file_path = osp.join(fig_path, f"{fig_path}_C{name[:-1]}_t{step}.pdf")
    plot_embedding(xy, lims=90, file_path=file_path)
    return acc1, acc2











