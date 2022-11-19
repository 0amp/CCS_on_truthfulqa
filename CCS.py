import torch as t
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class LinearProbe(nn.Module): 
    def __init__(self, d): 
        super().__init__()
        self.linear = nn.Linear(d,1)

    def reset(self): 
        self.linear.reset_parameters()

    def forward(self, x): 
        return t.sigmoid(self.linear(x))

class CCS(): 
    def __init__(self, x0, x1, y, nepochs = 1000, ntries = 10, lr = 1e-1, device='cpu'):
        # x0, x1 are activations for training data, y is labels
        self.x0 = x0
        self.x1 = x1
        self.d = self.x0.shape[-1] # dim of activations
        self.y = y

        # probe
        self.probe = LinearProbe(self.d)
        self.flag = 'acc' # whether to use acc or 1-acc

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.device = device

    def get_flag(self): 
        return self.flag
    
    def get_loss(self, p0, p1): 
        """
        Returns the CCS loss for two probability tensors each of shape (n,1) or (n,)
        """
        informative_loss = (t.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        prob_loss = ((1-(p0 + p1))**2).mean(0)
        return informative_loss + consistent_loss + prob_loss
    
    def get_acc(self, probe): 
        """
        Computes accuracy for the current parameters on all data
        """
        p0, p1 = probe(self.x0), probe(self.x1)
        decode = 0.5 * (p0 + (1-p1))
        pred = t.Tensor([0 if d > 0.5 else random.randint(0,1) if d == 0.5 
                         else 1 for d in decode])
        acc = (pred == self.y).float().mean().item()
        if probe == self.probe: 
            flag = 'acc' if acc > 1-acc else '1-acc'
        return max(acc, 1-acc)
    
    def make_pred(self, x): 
        if self.flag == 'acc': 
            return self.probe(x).detach().cpu().numpy()
        else: 
            return (1-self.probe(x)).detach().cpu().numpy()

    def pred_acc(self, x0, x1, y): 
        """
        Computes accuracy for current parameters on x0, x1, y
        """
        p0, p1 = self.probe(x0), self.probe(x1)
        decode = 0.5 * (p0 + (1-p1))
        pred = t.Tensor([0 if d > 0.5 else random.randint(0,1) if d == 0.5 
                         else 1 for d in decode])
        acc = (pred == y).float().mean().item()
        if self.flag == 'acc': 
            return acc
        else: 
            return 1-acc
    
    def train(self): 
        """
        Does a single training run of nepochs epochs
        """
        best_acc = self.get_acc(self.probe)
        best_loss = 1e4

        for _ in tqdm(range(self.ntries)): 
            probe = LinearProbe(self.d)
            optimizer = t.optim.AdamW(probe.parameters(), lr=self.lr)

            for epoch in range(self.nepochs): 
                p0, p1 = probe(self.x0), probe(self.x1)
                optimizer.zero_grad()
                loss = self.get_loss(p0,p1)
                loss.backward()
                optimizer.step()

                loss_np = loss.detach().item()

            acc = self.get_acc(probe)

            if acc > best_acc: 
                self.probe = probe

                best_acc = acc
                best_loss = loss_np

        return best_acc, best_loss