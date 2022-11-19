# %% [markdown]
# Setup

# %%
%load_ext autoreload
%autoreload 2

# %%
import torch as t
import pandas as pd
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from CCS import CCS
from utils import *

# %%
data = pd.read_csv('data/modifiedtqa.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'Yes' else 0)
data['yes'] = data['question'] + ' Yes'
data['no'] = data['question'] + ' No'
data = data.drop(['question'], axis=1)
data = data.to_dict('records')
data = data[:200]

# %%
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
config = GPT2Config.from_pretrained("gpt2-xl", output_hidden_states=True)
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=config) 

def generate(prompt, max_length = 40, do_sample=True, top_p = 0.95, top_k =60, **model_kwargs): 
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    output_sequences = model.generate(inputs,max_length=max_length, 
                                      do_sample=do_sample,top_p=top_p,
                                      top_k=top_k,**model_kwargs)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

out = generate(data[0]['yes'])
print(out)

# %%
activations, labels = get_activations(model, tokenizer, data, [24], 1)
print(activations.shape)
labels.shape

# %% 
# get activations from .pt file
acts = t.load('activations/gpt2-xl/activations.pt')
acts = acts[:200]
# %%
x0 = activations[:100,0,:].detach()
x0 = (x0 - x0.mean(axis=0, keepdims=True))/x0.std(axis=0, keepdims=True)
x1 = activations[:100,1,:].detach()
x1 = (x1 - x1.mean(axis=0, keepdims=True))/x1.std(axis=0, keepdims=True)
y = labels[:100]
testx0 = activations[100:,0,:].detach()
testx0 = (testx0 - testx0.mean(axis=0, keepdims=True))/testx0.std(axis=0, keepdims=True)
testx1 = activations[100:,1,:].detach()
testx1 = (testx1 - testx1.mean(axis=0, keepdims=True))/testx1.std(axis=0, keepdims=True)
testy = labels[100:]

ccs = CCS(x0, x1, y, ntries=10)
print("\n", ccs.train())
print(ccs.get_flag())

# WEIRD THINGS
## back to always having the flag be 'acc' even though that leads to bad pred scores
## want the p0 and p1 to sum to 1 and working on a loss term to enforce that
## add MLP probe back in to see if that helps

# %%
print(ccs.pred_acc(testx0, testx1, testy))

# %%
def view_CCS_pred(data_index, ccs): 
    """
    View CCS predictions for both answer choices for dataset[data_index]
    """
    prmpt1 = data[i]['yes']
    prmpt2 = data[i]['no']
    x0 = get_one_activation(model, tokenizer, prmpt1, [8]).unsqueeze(0).detach()
    x1 = get_one_activation(model, tokenizer, prmpt2, [8]).unsqueeze(0).detach()
    y = t.tensor([data[i]['label']])
    print(f'{ccs.probe(x0).item()}: {prmpt1}')
    print(f'{ccs.probe(x1).item()}: {prmpt2}')
    print(ccs.pred_acc(x0, x1, y))

for i in range(10): 
    view_CCS_pred(i, ccs)

# %%
# RESULT: 0.5312

print(get_normalized_score(model, tokenizer, data, labels))
        