import numpy as np
from typing import List, Tuple
import os
from dotenv import load_dotenv
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from datetime import date

import openai

from transformers import T5Tokenizer, T5ForConditionalGeneration

from model_wrappers import HFModelWrapper, OpenAIModel
from probes import CCS, TPC, LR

class ELK():
    def __init__(self, model_name, use_cuda = True):
        """

        """
        
        if "gpt" in model_name:
            self.mt = HFModelWrapper(model_name, use_cuda = use_cuda)
            
        elif "ada" in model_name or "babbage" in model_name or "curie" in model_name:
            load_dotenv()

            openai.api_key = os.getenv("OPENAI_API_KEY")

            self.mt = OpenAIModel(model_name)
        elif "t5" in model_name:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            self.mt = HFModelWrapper(model_name, tokenizer = tokenizer, model = model, use_cuda = use_cuda)
                    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.CCS = CCS()
        self.TPC = TPC()
        self.LR = LR()

    def gen_hidden_states(self, yes_examples: List[str], no_examples: List[str], layers: List[int], 
                          store_acts: bool = False,
                          dataset_name: str = ""): 
        """
        Generates hidden states at specified layers for yes and no sentence pairs. 
        Output shape is (num_examples, len(layers), hidden_size). Inputs are a list
        of strings. 
        """

        # yes_examples = [example for example in yes_examples if len(self.mt.tokenizer(example).input_ids) < self.mt.n_ctx] #this was causing cuda errors w t5 for some reason
        # no_examples = [example for example in no_examples if len(self.mt.tokenizer(example).input_ids) < self.mt.n_ctx]

        x_plus_acts, x_minus_acts = self.mt.get_activations_last_idx(yes_examples, layers), self.mt.get_activations_last_idx(no_examples, layers)
        
        if store_acts:
            torch.save(x_plus_acts, f"activations/{self.mt.model_name}/{dataset_name}_x_plus_activations_{date.today()}.pt")
            torch.save(x_minus_acts, f"activations/{self.mt.model_name}/{dataset_name}_x_minus_activations_{date.today()}.pt")

        return x_plus_acts, x_minus_acts
    
    def train_probe(self, yes_acts, no_acts, labels = None, probe_type="CCS"): 
        if probe_type == "CCS":
            return self.CCS.fit(yes_acts, no_acts)
        elif probe_type == "LR":
            self.LR.fit(yes_acts, no_acts, labels)
        elif probe_type == "TPC":
            self.TPC.fit(yes_acts, no_acts)
        else: 
            raise
    
    def score_probe(self, yes_acts, no_acts, labels, probe_type="CCS"):     
        if probe_type == "CCS":
            return self.CCS.score(yes_acts, no_acts, labels)
        elif probe_type == "TPC":
            return self.TPC.score(yes_acts, no_acts, labels)
        elif probe_type == "LR":
            return self.LR.score(yes_acts, no_acts, labels)
        else:
            raise

    def predict_probe(self, yes_acts, no_acts, probe_type="CCS"):
        if probe_type == "CCS":
            return self.CCS.predict(yes_acts, no_acts)
        elif probe_type == "TPC":
            return self.TPC.predict(yes_acts, no_acts)
        elif probe_type == "LR":
            return self.LR.predict(yes_acts, no_acts)
        else:
            raise

    def test_probe_outputs(self, yes_examples, no_examples, labels, layers, probe_types=["CCS"]): 
        """
        Test probe performance on yes / no examples by printing out input sentence 
        and probe output. Inputs are a list of strings. 
        """
        yes_tokens = self.mt.tokenize(yes_examples) # TODO make sure .tokenize works for all models
        no_tokens = self.mt.tokenize(no_examples)
        yes_acts, no_acts = self.gen_hidden_states(yes_tokens, no_tokens, 1, layers)
        preds = []
        for probe in probe_types: 
            preds.append(self.predict_probe(yes_acts, no_acts, probe_type=probe))
        for i in range(len(yes_examples)): 
            print("PROMPT: ", yes_examples[i])
            for i, probe in enumerate(probe_types):
                print(probe, ": ", preds[i][i])
            print("LABEL: ", labels[i])
            print("---------")

    def normalized_zero_shot_prob(self, prompts, dataset_name): 
        """
        Test zero shot performance of model on a set of yes ands by 
        comparing normalized scores of the 'yes' and 'no' logits. Inputs are a
        list of strings. 
        """
        label_dict = {
            "imdb": ["negative", "positive"], # This is for normal IMDB
            "amazon-polarity": ["negative", "positive"],
            "ag-news": ["politics", "sports", "business", "technology"],
            "modifiedtqa": ["Yes", "No"],
        }
        
        # TODO: assumes get_logit is implemented for all models
        yes_logits = torch.stack([self.mt.get_logit(prompt, label_dict[dataset_name][0]) for prompt in tqdm(prompts)])
        no_logits = torch.stack([self.mt.get_logit(prompt, label_dict[dataset_name][1]) for prompt in tqdm(prompts)])
        stacked = torch.stack([yes_logits, no_logits], dim=1) #n x 2
        return F.softmax(stacked, dim=1).cpu().detach().numpy()

    def zero_shot_score(self, prompts, labels, dataset_name): 
        zero_shot_prob = self.normalized_zero_shot_prob(prompts, dataset_name)
        
        if dataset_name == "modifiedtqa":
            return ((zero_shot_prob[:, 0] > 0.5).astype(int) == labels).astype(int).sum() / zero_shot_prob.shape[0]
        else: 
            #if it's positive, zero_shot_prob is 1, which lines up with labels
            return ((zero_shot_prob[:, 1] > 0.5).astype(int) == labels).astype(int).sum() / zero_shot_prob.shape[0]



    def finetune(self, dataset, batch_size, epochs, lr, layers, probe_type="CCS"): 
        """
        Run model on the given dataset (list of tuples of strings?), collect probe 
        outputs, and finetune the model's output yes/no logit distribution on the 
        probe's yes/no distribution. Assumes probe is already trained. 
        """

        optimizer = torch.optim.Adam(self.mt.model.parameters(), lr=lr)
        self.mt.model.train()

        for epoch in range(epochs): 
            torch.random.shuffle(dataset) # does that work? 
            for i in range(0, len(dataset), batch_size): 
                batch = dataset[i:i+batch_size]

                # get probe output
                yes_examples = [x[0] + " yes" for x in batch]
                no_examples = [x[0] + " no" for x in batch]
                yes_tokens = self.mt.tokenize(yes_examples)
                no_tokens = self.mt.tokenize(no_examples)
                yes_acts, no_acts = self.gen_hidden_states(yes_tokens, no_tokens, 1, layers)
                probe_out = self.probe_predict(yes_acts, no_acts, probe_type)

                # get model output
                model_out = self.normalized_zero_shot_prob([x[0] for x in batch], [x[1] for x in batch])

                # finetune model
                loss = F.cross_entropy(model_out, probe_out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print("EPOCH: ", epoch, ", LOSS: ", loss)
        
        self.mt.model.eval()
                

    

