import os
import functools
import argparse
import copy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset

from abc import ABC, abstractmethod
 
class Probe(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def score(self):
        pass
    
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

def normalize(x: torch.Tensor, var_normalize: bool = False):
    """
    Mean-normalizes the data x (of shape (n, d))
    If self.var_normalize, also divides by the standard deviation
    """
    normalized_x = x - x.mean(axis=0, keepdims=True)
    if var_normalize:
        normalized_x /= normalized_x.std(axis=0, keepdims=True)

    return normalized_x

def normalize_then_diff(x0: torch.Tensor, 
                        x1: torch.Tensor, 
                        var_normalize: bool = True):        
    return normalize(x0, var_normalize=var_normalize) - normalize(x1, var_normalize=var_normalize)
    
class CCS(Probe):
    def __init__(self, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
 
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Linear(self.d, 1)
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def tensorize_data(self, x):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        return torch.tensor(x, dtype=torch.float, requires_grad=False, device=self.device)

    def score(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        predictions = self.predict(x0_test, x1_test)

        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc
    
    def predict(self, x0, x1):
        x0 = normalize(self.tensorize_data(x0), var_normalize = self.var_normalize) #? what happens when there's only one datapoint
        x1 = normalize(self.tensorize_data(x1), var_normalize = self.var_normalize)
        
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        
        return predictions

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.x0.copy(), self.x1.copy() #these are already normalized
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
            # probe
            p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.detach().cpu().item()
    
    def fit(self, x0, x1):
        self.x0 = normalize(self.tensorize_data(x0), var_normalize = self.var_normalize)
        self.x1 = normalize(self.tensorize_data(x1), var_normalize = self.var_normalize)
        self.d = self.x0.shape[-1]
        
        best_loss = np.inf
        for train_num in tqdm(range(self.ntries)):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss


class TPC(Probe):
    def __init__(self, n_components = 5, svd_solver="full"):
        self.model = PCA(n_components=n_components, svd_solver=svd_solver)
    
    def fit(self, x0, x1):  
        #n x n_layer x dim
        data = normalize_then_diff(x0, x1).cpu().numpy().reshape(-1, x0.shape[-1])
        self.model.fit(data)
        
    def predict(self, x0, x1):
        data = normalize_then_diff(x0, x1).cpu().numpy().reshape(-1, x0.shape[-1])
        
        return ((data @ self.model.components_[0]) < 0.5).astype(int)
    
        # return self.model.transform(normalize_then_diff(x0, x1).cpu().numpy())
    
    def score(self, x0, x1, labels):
        predictions = self.predict(x0, x1)
        
        acc = (predictions == labels.numpy()).mean()
        acc = max(acc, 1 - acc)

        return acc
        # return self.model.score(data, labels)
        
class LR(Probe):
    def __init__(self):
        self.model = LogisticRegression(max_iter = 10_000, n_jobs = 1, C = 0.1, class_weight="balanced")

    def fit(self, x0, x1, labels):
        data = (x0 - x1).reshape(-1, x0.shape[-1])
        # data = normalize_then_diff(x0, x1).cpu().numpy().reshape(-1, x0.shape[-1])
        
        self.model.fit(data, labels)
    
    def predict(self, x0, x1):
        data = (x0 - x1).reshape(-1, x0.shape[-1])

        # data = normalize_then_diff(x0, x1).cpu().numpy().reshape(-1, x0.shape[-1])
        
        self.model.predict(data)
    
    def score(self, x0, x1, labels):
        data = (x0 - x1).reshape(-1, x0.shape[-1])

        # data = normalize_then_diff(x0, x1).cpu().numpy().reshape(-1, x0.shape[-1])
        
        return self.model.score(data, labels)