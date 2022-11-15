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


def uap_sgd(model, data_loader, nb_epoch, eps=10./255., beta=10, step_decay=0.9,
            y_target=None, loss_fn=None, layer_name=None,
            uap_init=None, device=torch.device("cuda:1"), steps=500,):
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
    _, (x_val, y_val) = next(enumerate(data_loader))
    batch_size = len(x_val)
    if uap_init is None:
        batch_delta = torch.zeros_like(x_val)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1]).detach()
    losses = []
    delta = batch_delta[0]

    # loss function
    if layer_name is None:
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.tensor([beta], device=device, dtype=torch.float64)

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

    # layer maximization attack
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = torch.norm(forward_output, p='fro')

        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)

    model.to(device)
    batch_delta.requires_grad = True
    for epoch in range(nb_epoch):
        epoch_losses = []
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay
        phar = tqdm(enumerate(data_loader))

        loader = iter(data_loader)
        for i in range(steps):
            try:
                x_val, y_val = next(loader)
            except:
                loader = iter(data_loader)
                x_val, y_val = next(loader)
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            # for targeted UAP, switch output labels to y_target
            if y_target is not None: y_val = torch.ones(size=y_val.shape, dtype=y_val.dtype, device=device) * y_target
            perturbed = (x_val + batch_delta).to(device)
            outputs = model(perturbed)

            # loss function value
            if layer_name is None:
                loss = clamped_loss(outputs, y_val)
            else:
                loss = main_value
            if y_target is not None: loss = -loss  # minimize loss for targeted UAP
            losses.append(torch.mean(loss))
            epoch_losses.append(torch.mean(loss))
            loss.backward()

            # batch update
            grad_sign = batch_delta.grad.data.mean(dim=0).sign()
            delta = delta + grad_sign * eps_step
            batch_delta.grad.data.zero_()

            preds = outputs.argmax(dim=1).detach().cpu()
            phar.set_description(f"-> [Train] UAP:{y_target} epcoh:{epoch} y_target:{y_target} loss:{torch.mean(loss).item()} pred:{preds[:10]}")

    if layer_name is not None: handle.remove()  # release hook
    return delta.data, losses