import os
import functools
import argparse
import copy

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset

label_dict = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon-polarity": ["negative", "positive"],
    "ag-news": ["politics", "sports", "business", "technology"],
}

hf_label_map = {
    "imdb": {
        0: "positive",
        1: "negative",
    },
    "amazon-polarity": {
        0: "Negative",
        1: "Positive"
    }, 
    "ag-news": {
        0: "World Politics",
        1: "Sports",
        2: "Business",
        3: "Science and Technology"
    }
}

hf_label_key = {
    "imdb": ["label", "text"]
}

prompt_dict = {
    "imdb": [
        "Consider the following example: '''[text]'''\nBetween positive and negative, the sentiment of this example is [label]",
        "Consider the following example: '''[text]'''\nBetween positive and negative, which is the sentiment of this example? [label]",
        # "Between positive and negative, the following movie review expresses a [label] sentiment:\n [text]",
    ],
    "amazon-polarity": [
        '[prefix]Consider the following example: " [text] ” Between positive and negative, the sentiment of this example is [label]',
        '[prefix]Consider the following example: “ [text] ” Between positive and negative, which is the sentiment of this example? [label]',
    ],
    "ag-news": [
        '[prefix]Consider the following example: “ [text] ” Choice 1: [label0]. Choice 2: [label1]. Between choice 1 and choice 2, the topic of this example is [label]',
        '[prefix]Consider the following example: “ [text] ” Choice 1: [label0]. Choice 2: [label1]. Between choice 1 and choice 2, what is the topic of this example? [label]',
        '[prefix]Consider the following example: “ [text] ” Which is the topic of this example, choice 1: [label0], or choice 2: [label1]? [label]',
        '[prefix][text] Choice 1: [label0]. Choice 2: [label1]. Between choice 1 and choice 2, the topic of this example is [label]',
        '[prefix][text] Choice 1: [label0]. Choice 2: [label1]. Between choice 1 and choice 2, what is the topic of this example? [label]',
        '[prefix][text] Which is the topic of this example, choice 1: [label0], or choice 2: [label1]? [label]',
        '[prefix][text] What label best describes this news article, choice 1: [label0], or choice 2: [label1]? [label]',
        '[prefix][text] Which section of a newspaper would this article likely appear in, choice 1: [label0], or choice 2: [label1]? [label]',
    ],
}

def multiple_replace(dict, text):
    #from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

class Prompts():
    def __init__(self, dataset_name: str, N: int = 500, tokenizer_name:str = "gpt2", prefix: str = "", max_len: int = 1024, random = False):
        if not random:
            np.random.seed(0)
            
        self.dataset_name = dataset_name
        self.N = N
        
        if dataset_name == "modifiedtqa":
            modifiedtqa = pd.read_csv('data/modifiedtqa.csv').iloc[:N]
            
            modifiedtqa['label'] = modifiedtqa['label'].apply(lambda x: 1 if x == 'Yes' else 0)
            
            modifiedtqa.rename(columns = {"question": "text"}, inplace=True)
            
            modifiedtqa['x_plus'] = modifiedtqa['text'] + ' Yes'
            modifiedtqa['x_minus'] = modifiedtqa['text'] + ' No'
            
            modifiedtqa['x_plus_true'] = modifiedtqa['label']
            
            self.dataset = modifiedtqa
            
        else:
            dataset = load_dataset(dataset_name)['train']
                        
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            sents = []
            true_labels = []
            for _ in tqdm(range(N)):
            # for simplicity, sample a random example until we find one that's a reasonable length
            # (most examples should be a reasonable length, so this is just to make sure)
                while True:
                    idx = np.random.randint(len(dataset))
                    text, true_label = dataset[idx]["content" if self.dataset_name == "amazon-polarity" else "text"], dataset[idx]["label"]
                    # the actual formatted input will be longer, so include a bit of a marign
                    if len(tokenizer(text).input_ids) < max_len - 100:  
                        # print( len(tokenizer(text).input_ids))
                        break
                
                sents.append(text)
                true_labels.append(true_label)

            texts = {
                "text": sents,
                "label": true_labels,
            }
            
            # short_text_indices = [i for i, text in enumerate(dataset['train']["content" if self.dataset_name == "amazon-polarity" else "text"][:N*2]) if len(tokenizer(text).input_ids) < max_len - 100]
            
            # indices = np.random.randint(0, len(short_text), size = N)
            # texts = dataset['train'][np.random.choice(short_text_indices, size = N, replace = False)]
            
            print(sum(texts['label']))
            
            x_pluses = []
            x_minuses = []
            x_pluses_true = []
            for label, text in zip(texts["label"], texts["content" if self.dataset_name == "amazon-polarity" else "text"]):
                prompt_format = prompt_dict[self.dataset_name][np.random.randint(0, len(prompt_dict[self.dataset_name]))] #get a random prompt format

                # true_label = hf_label_map[self.dataset_name][label] 

                # false_labels = list(hf_label_map[self.dataset_name].keys())
                # false_labels.remove(label) #remove the hf true label
                # false_label = hf_label_map[self.dataset_name][false_labels[np.random.randint(0, len(false_labels))]]
                
                # if np.random.rand() < 0.5:
                #     label0 = true_label
                #     label1 = false_label
                # else:
                #     label0 = false_label
                #     label1 = true_label
                
                replace_dict = {
                    "[prefix]" : prefix,
                    "[text]": text,
                    # "[label0]": label0,
                    # "[label1]": label1,
                    "[label]": hf_label_map[self.dataset_name][0] 
                }

                
                # x_pluses_true.append(True if label == true_label else False)
                    
                x_pluses.append(multiple_replace(replace_dict, prompt_format))

                replace_dict["[label]"] = hf_label_map[self.dataset_name][1] 
                x_minuses.append(multiple_replace(replace_dict, prompt_format))
            
            texts['x_plus'] = x_pluses
            texts['x_minus'] = x_minuses
            # texts['x_plus_true'] = x_pluses_true
            
            self.dataset = pd.DataFrame.from_dict(texts)
    
    def gen_train_test_indices(self, train_ratio: float = 0.6, test_ratio: float = 0.4, set_instance_vars = False):
        
        train_indices = np.random.randint(0, len(self.dataset.index), size = int(self.N * train_ratio))
        test_indices = list(set(range(self.N)) - set(train_indices))
        
        if set_instance_vars:
            self.train = self.dataset.iloc[train_indices]
            self.test = self.dataset.iloc[test_indices]
        
        return train_indices, test_indices
            
                

            
