import math
import torch
import torch.nn as nn
from tqdm import tqdm
'''
Basic version of untargeted stochastic gradient descent UAP adapted from:
[AAAI 2020] Universal Adversarial Training
- https://ojs.aaai.org//index.php/AAAI/article/view/6017

Layer maximization attack from:
Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training
- https://arxiv.org/abs/1911.10364
'''


def uap_sgd(model, data_loader, nb_epoch, eps=5e-3, beta=1.0, step_decay=0.9,
            y_target=None, loss_fn=None, layer_name=None,
            uap_init=None, device=torch.device("cuda:0")):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})

    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    model.eval()
    model.to(device)
    layer_loss = torch.zeros([1]).to(device)

    _, (x_val, y_val) = next(enumerate(data_loader))
    batch_size = len(x_val)
    if uap_init is None:
        batch_delta = torch.zeros_like(x_val)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1]).detach()
    losses = []
    delta = batch_delta[0]

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    beta = torch.tensor([beta], device=device, dtype=torch.float64)

    def clamped_loss(output, target):
        loss = torch.mean(torch.min(loss_fn(output, target), beta))
        return loss

    # layer maximization attack
    if layer_name is not None:
        def get_norm(self, forward_input, forward_output):
            global layer_loss
            layer_loss = torch.norm(forward_output, p='fro')

        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)

    batch_delta.requires_grad = True
    nb_steps = int(nb_epoch * len(data_loader))
    for epoch in range(nb_epoch):
        epoch_losses = []
        phar = tqdm(enumerate(data_loader))
        loader = iter(data_loader)
        for step in range(len(data_loader)):
            try:
                x_val, y_val = next(loader)
            except:
                loader = iter(data_loader)
                x_val, y_val = next(loader)

            # perturbation step size with decay
            eps_step = eps * math.pow(step_decay, 1.0 * (step + epoch * len(data_loader)) / nb_steps)
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])

            # for targeted UAP, switch output labels to y_target
            if y_target is not None:
                y_val = torch.ones(size=y_val.shape, dtype=y_val.dtype, device=device) * y_target
            perturbed = (x_val + batch_delta).to(device)
            outputs = model(perturbed)
            ce_loss = clamped_loss(outputs, y_val)

            if layer_name is None:
                loss = ce_loss
            else:
                ce_loss = clamped_loss(outputs, y_val)
                loss = ce_loss - layer_loss.mean()

            if y_target is not None: loss = -loss  # minimize loss for targeted UAP
            losses.append(torch.mean(loss))
            epoch_losses.append(torch.mean(loss))
            loss.backward()

            # batch update
            grad_sign = batch_delta.grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * eps_step
            batch_delta.grad.data.zero_()
            preds = outputs.argmax(dim=1).detach()
            accuracy = 100.0 * (y_val.eq(preds).sum() / len(preds))

            phar.set_description(
                f"-> [Train] UAP:{y_target} E:[{epoch+1}/{nb_epoch}] y_target:{y_target} "
                f"loss:{round(float(loss.data), 4)}=ce_loss:{round(float(ce_loss.data), 4)}+layer_loss:{round(float(layer_loss.mean().data), 4)} "
                f"pred:{preds[:5]} acc:{accuracy}%")

            # increase learning rate
            if accuracy < 5.0:
                eps = eps * 1.02
            if loss < 1e-5:
                break

    if layer_name is not None: handle.remove()  # release hook
    return delta.data, losses