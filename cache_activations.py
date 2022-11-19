import torch as t
import pandas as pd
from tqdm import tqdm
from einops import rearrange, reduce, repeat

data = pd.read_csv('data/modifiedtqa.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'Yes' else 0)
data['yes'] = data['question'] + ' Yes'
data['no'] = data['question'] + ' No'
data = data.drop(['question'], axis=1)
data = data.to_dict()
data['yes'] = list(data['yes'].values())
data['no'] = list(data['no'].values())
data['label'] = list(data['label'].values())

from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2-xl", output_hidden_states=True)
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=config) 

batch_size = 100

dlen = len(data['yes'])
labels = t.tensor(data['label'][:])
batched_yes = [data['yes'][i:i+batch_size] for i in range(0, dlen, batch_size)]
batched_no = [data['no'][i:i+batch_size] for i in range(0, dlen, batch_size)]
# tokenize batches
batched_yes = [tokenizer(batch, return_tensors="pt", padding=True)["input_ids"] for batch in tqdm(batched_yes)]
batched_no = [tokenizer(batch, return_tensors="pt", padding=True)["input_ids"] for batch in tqdm(batched_no)]
# run batch through model
acts = t.cat([t.stack([t.stack(model(batched_yes[b])[2])[:,:,-1], t.stack(model(batched_no[b])[2])[:,:,-1]]) for b in tqdm(range(len(batched_yes)))], dim=2)
acts = rearrange(acts, 'y l b a -> b y l a')
acts = acts.cpu().detach()
print(acts.shape)

# cache activations somewhere
t.save(acts, 'activations/gpt2-xl/activations.pt')