#%%
%load_ext autoreload
%autoreload 2

from tqdm import tqdm
from einops import rearrange, reduce, repeat
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm import tqdm
from einops import rearrange

from model_wrappers import HFModelWrapper
from elk import ELK

# %%
data = pd.read_csv('data/modifiedtqa.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'Yes' else 0)
data['yes'] = data['question'] + ' Yes'
data['no'] = data['question'] + ' No'
# data = data.drop(['question'], axis=1)

data.head()

#%%
data = data.to_dict() #745 examples
data['yes'] = list(data['yes'].values())
data['no'] = list(data['no'].values())
data['label'] = list(data['label'].values())

#%%
elk = ELK("gpt2")
yes_acts, no_acts = elk.gen_hidden_states(data['yes'], data['no'], [12])

#%%
elk.train_probe(yes_acts, no_acts, probe_type = "CCS")

#%%
elk.score_probe(yes_acts, no_acts, torch.tensor(data['label']))

#%%
probs = elk.normalized_zero_shot_prob(data['question'].tolist())

#%%

elk.zero_shot_score(data['question'].tolist(), data['label'])


#%%
elk.train_probe(yes_acts, no_acts, labels = data['label'], probe_type = "LR")

#%%

#%%
torch.save(acts, f'activations/{model_name}/activations.pt')
