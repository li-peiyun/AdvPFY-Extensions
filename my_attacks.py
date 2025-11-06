# my_attacks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# FGSM (L_inf)
def fgsm_linf(input_img, target, model, eps=0.2, obj_func=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if obj_func is None:
        obj_func = nn.CrossEntropyLoss()
    delta = torch.zeros_like(input_img, requires_grad=True).to(device)
    model.zero_grad()
    y_pred = model(input_img + delta)
    loss = obj_func(y_pred, target)
    loss.backward()
    delta.data = (eps * torch.sign(delta.grad.data)).clamp(-eps, eps)
    adv_img = torch.clamp(input_img + delta.detach(), 0, 1)
    return adv_img.detach()


# BIM / Iterative FGSM (like PGD without random start)
def bim_linf(input_img, target, model, atk_itr=10, eps=0.2, alpha=1/255, obj_func=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if obj_func is None:
        obj_func = nn.CrossEntropyLoss()
    delta = torch.zeros_like(input_img, requires_grad=True).to(device)
    for _ in range(atk_itr):
        model.zero_grad()
        y_pred = model(input_img + delta)
        loss = obj_func(y_pred, target)
        loss.backward()
        delta.data = (delta + alpha * torch.sign(delta.grad.data)).clamp(-eps, eps)
        delta.data = torch.clamp(input_img + delta, 0, 1) - input_img
        delta.grad.zero_()
    adv_img = torch.clamp(input_img + delta.detach(), 0, 1)
    return adv_img.detach()


# FGSM with L2 projection (approx)
def fgsm_l2(input_img, target, model, eps=0.2, obj_func=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if obj_func is None:
        obj_func = nn.CrossEntropyLoss()
    delta = torch.zeros_like(input_img, requires_grad=True).to(device)
    model.zero_grad()
    y_pred = model(input_img + delta)
    loss = obj_func(y_pred, target)
    loss.backward()
    grad = delta.grad.data
    # per-sample L2 norm
    grad_flat = grad.view(grad.shape[0], -1)
    grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1,1,1,1).clamp_min(1e-8)
    delta.data = (eps * grad / grad_norm).clamp(-eps, eps)
    adv_img = torch.clamp(input_img + delta.detach(), 0, 1)
    return adv_img.detach()


# PGD with random start wrapper (uses same step update as pgd_linf)
def pgd_linf_rs(input_img, target, model, atk_itr=10, eps=0.2, alpha=1/255, obj_func=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if obj_func is None:
        obj_func = nn.CrossEntropyLoss()
    # random start in [-eps, eps]
    delta = ((torch.rand_like(input_img) - 0.5) * 2 * eps).to(device)
    delta.requires_grad_()
    for _ in range(atk_itr):
        model.zero_grad()
        y_pred = model(input_img + delta)
        loss = obj_func(y_pred, target)
        loss.backward()
        delta.data = (delta + alpha * torch.sign(delta.grad.data)).clamp(-eps, eps)
        delta.data = torch.clamp(input_img + delta, 0, 1) - input_img
        delta.grad.zero_()
    adv_img = torch.clamp(input_img + delta.detach(), 0, 1)
    return adv_img.detach()
