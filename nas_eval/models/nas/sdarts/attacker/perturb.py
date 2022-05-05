import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nas.sdarts.attacker.linf_sgd import Linf_SGD
from torch.optim import SGD, Adam


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD_alpha(model, X, y, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2*epsilon/steps)
    with torch.no_grad():
        loss_before = model._loss(X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()
        
    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss = -model._loss(X, y, updateType='weight')
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()
        
    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after = model._loss(X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()


def Random_alpha(model, X, y, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()


def Linf_PGD_alpha_RNN(model, X, y, hidden, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2*epsilon/steps)
    with torch.no_grad():
        loss_before, _ = model._loss(hidden, X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()

    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss, _ = model._loss(hidden, X, y, updateType='weight')
        loss = -loss
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()
    
    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after, _ = model._loss(hidden, X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()
    
        
def Random_alpha_RNN(model, X, y, hidden, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()




