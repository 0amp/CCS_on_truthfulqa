from typing import List, Optional, Tuple, Dict

import re
import numpy as np
from tqdm import tqdm 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

import openai

from utils import set_requires_grad, untuple

class OpenAIModel():
    def __init__(self, engine):
        self.engine = engine
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], engine=self.engine)['data'][0]['embedding']
    
class HFModelWrapper():

    def __init__(
        self, 
        model_name:str = None, 
        model:AutoModelForCausalLM = None, 
        tokenizer:AutoTokenizer = None, 
        low_cpu_mem_usage:bool = False,
        use_cuda:bool = False
    ):
        if tokenizer is None:
            assert model_name is not None
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        if model is None:
            assert model_name is not None

            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage
            )
            set_requires_grad(False, model)
            
        self.tokenizer = tokenizer
        self.model = model
        self.name = model_name
        
        if "gpt" in self.name:
            self.num_layers = len(
                [
                    n
                    for n, m in model.named_modules()
                    if re.match(r"^transformer\.h\.\d+$", n)
                ]
            )
        elif "t5" in self.name:
            self.num_layers = len(
                [
                    n for n, m in model.named_modules() 
                    if re.match(r'^encoder\.block\.\d$', n)
                ] + [
                    n for n, m in model.named_modules() 
                    if re.match(r'^decoder\.block\.\d$', n)
                ]
            )
        
        assert self.num_layers > 0
        
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"

        if self.device == "cpu":
            self.model.eval()
        else:
            self.model.eval().cuda()

        if use_cuda:
            self.model.parallelize()
    
    def get_logit(self, prompt, token):
        input = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        logits = self.model(input)[0][0,-1,:].detach()
        return logits[self.tokenizer(token).input_ids[0]]


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        return outputs
    
    def generate(self, prompt, num_tokens=100, temperature=1.0, top_k=0, top_p=0.9, repetition_penalty=1.0, do_sample=True, num_beams=1, early_stopping=False, use_cache=True, **model_kwargs):
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        model_logits = self.model.generate(input_ids, max_length=num_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample, num_beams=num_beams, early_stopping=early_stopping, use_cache=use_cache, **model_kwargs)
        return self.tokenizer.decode(model_logits[0], skip_special_tokens=True)

    def get_hidden_state(self, prompt: str, layers: List[int]):
        tokens = self.tokenizer.encode_plus(prompt, return_tensors='pt').to(self.device) #dict containing input_ids and attention_mask
        
        output = self.model(**tokens, output_hidden_states = True) #

        #hidden states has n_layers + 1 entires: one for the output of embeddings, one for output of each layer
        #with shape layer x batch x token x dim 
        return torch.stack(output['hidden_states'])[layers, :, -1, :].squeeze(1) #batch dim gets squeezed out
    
    def get_activations_last_idx(self, prompts: List[str], layers:List[int]):
        # tokens = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding = True).to(self.device) #dict containing input_ids and attention_mask
        #TODO: vectorize this
        return torch.stack([self.get_hidden_state(prompt, layers = layers) for prompt in tqdm(prompts)]) #n x n_layers x dim

    def get_activations_batch(self, prompts, layer=0, device='cuda'):
        """
        Get activations for a batch of prompts. Output shape is (batch_size, seq_len, hidden_size)
        """
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        with t.no_grad():
            output = self.model(input_ids)
            return output[2][layer].cpu()
        
        
    # def train_CCS(self, x0, x1, nepochs=100, device='cuda'):
    #     self.ccs = CCS(x0, x1)
    #     print(self.ccs.repeated_train())

    # def CCS_pred_acc(self, x0, x1, y):
    #     return self.ccs.get_acc(x0, x1, y)

    # def get_CCS_acc(self, x0, x1, y): 
    #     return self.ccs.pred_acc(x0, x1, y)
    
    # def get_CCS_pred(self, x): 
    #     return self.ccs.make_pred(x)