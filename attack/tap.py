import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def tap_sgd(model1, model2, x, y, lr=0.1, step_decay=0.9, steps=40, device=torch.device("cuda:0")):
    model1.to(device)
    model2.to(device)
    batch_size = len(x)

    x = x.to(device)
    logits = F.softmax(model2(x), dim=1)
    prob, idx = torch.topk(logits, k=2, dim=1)
    top1_y = idx[:, 0].clone().detach().requires_grad_(False)
    top2_y = idx[:, 1].clone().detach().requires_grad_(False)

    best_list = []
    for idx in range(batch_size):
        min_dist = 100.0
        best_x = None
        adv_x = x[idx].unsqueeze(0)
        adv_x.requires_grad = True
        adv_x = adv_x.to(device)
        ben_y = torch.tensor([top1_y[idx]], device=device, dtype=torch.int64)
        adv_y = torch.tensor([top2_y[idx]], device=device, dtype=torch.int64)
        for step in range(steps):
            step_lr = lr * step_decay
            adv_x = adv_x.detach()
            adv_x.requires_grad = True

            z1 = model1(adv_x)
            z2 = model2(adv_x)
            loss_ce = F.cross_entropy(z1, ben_y) + F.cross_entropy(z2, adv_y)
            loss_margin = F.mse_loss(z1, z2, reduction="mean")
            loss = loss_ce + loss_margin

            pred1 = z1.argmax(dim=1)
            pred2 = z2.argmax(dim=1)
            if pred1 == ben_y and pred2 == adv_y and loss_margin < min_dist:
                min_dist = loss_margin
                best_x = adv_x.clone().detach().cpu()

            grad = torch.autograd.grad(loss, [adv_x], retain_graph=False, create_graph=False)[0]
            adv_x = adv_x - step_lr * grad

        if best_x is not None:
            best_list.append(best_x)
            print(f"-> idx:{idx} pred1:{pred1.tolist()[0]} pred2:{pred2.tolist()[0]} dist:{min_dist} loss:{loss.item()} loss_ce:{loss_ce.item()} loss_margin:{loss_margin.item()}")

    if len(best_list):
        torch.cat(best_list).detach().cpu()
    return []


