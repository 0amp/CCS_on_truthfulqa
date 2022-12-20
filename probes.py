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

def normalize(x, var_normalize = False):
    """
    Mean-normalizes the data x (of shape (n, d))
    If self.var_normalize, also divides by the standard deviation
    """
    normalized_x = x - x.mean(axis=0, keepdims=True)
    if var_normalize:
        normalized_x /= normalized_x.std(axis=0, keepdims=True)

    return normalized_x

def normalize_then_diff(x0, x1, var_normalize = False):        
    return normalize(x0, var_normalize=var_normalize) - normalize(x1, var_normalize=var_normalize)
    
class CCS(Probe):
    def __init__(self, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        # self.x0 = normalize(x0, var_normalize = self.var_normalize)
        # self.x1 = normalize(x1, var_normalize = self.var_normalize)
        # self.d = self.x0.shape[-1]

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
        
        # self.probe = self.initialize_probe()
        # self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Linear(self.d, 1)
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    
        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def score(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        predictions = self.predict(x0_test, x1_test)

        acc = (predictions == y_test.numpy()).mean()
        acc = max(acc, 1 - acc)

        return acc
    
    def predict(self, x0, x1):
        x0 = torch.tensor(normalize(x0, var_normalize = self.var_normalize), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(normalize(x1, var_normalize = self.var_normalize), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        
        return predictions

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.x0, self.x1 #these are already normalized
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
        self.x0 = normalize(x0, var_normalize = self.var_normalize)
        self.x1 = normalize(x1, var_normalize = self.var_normalize)
        self.d = self.x0.shape[-1]
        
        best_loss = np.inf
        for train_num in range(self.ntries):
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
        self.model.fit(normalize_then_diff(x0, x1))
        
    def predict(self, x0, x1):
        return self.model.transform(normalize_then_diff(x0, x1))
    
    def score(self, x0, x1, labels):
        return self.model.score(normalize_then_diff(x0, x1), labels)
        
class LR(Probe):
    def __init__(self):
        self.model = LogisticRegression(max_iter = 10_000, n_jobs = 1, C = 0.1)

    def fit(self, x0, x1, labels):
        self.model.fit(normalize_then_diff(x0, x1), labels)
    
    def predict(self, x0, x1):
        self.model.predict(normalize_then_diff(x0, x1))
    
    def score(self, x0, x1, labels):
        return self.model.score(normalize_then_diff(x0, x1), labels)