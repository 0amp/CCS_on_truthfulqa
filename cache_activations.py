
# %%
%load_ext autoreload
%autoreload 2

#%%
import torch as t
import pandas as pd
from tqdm import tqdm
from einops import rearrange

t.cuda.empty_cache()

data = pd.read_csv('data/modifiedtqa.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'Yes' else 0)
data['yes'] = data['question'] + ' Yes'
data['no'] = data['question'] + ' No'
data = data.drop(['question'], axis=1)

data.head()
#%%

data = data.to_dict()
data['yes'] = list(data['yes'].values())
data['no'] = list(data['no'].values())
data['label'] = list(data['label'].values())

#%%
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

model_name = "gpt2-medium"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained(model_name, output_hidden_states=True)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config) 

# make sure tokenizer and model are on the same device
model.eval()

model.cuda()
print("done")
#%%
batch_size = 20
device = "cuda"

dlen = len(data['yes'])
labels = t.tensor(data['label'][:])
batched_yes = [data['yes'][i:i+batch_size] for i in range(0, dlen, batch_size)]
batched_no = [data['no'][i:i+batch_size] for i in range(0, dlen, batch_size)]
# tokenize batches
batched_yes = [tokenizer(batch, return_tensors="pt", padding=True)["input_ids"].to(device) for batch in tqdm(batched_yes)]
batched_no = [tokenizer(batch, return_tensors="pt", padding=True)["input_ids"].to(device) for batch in tqdm(batched_no)]
# run batch through model
acts = t.tensor([])
for i in tqdm(range(len(batched_yes))): 
    y = t.stack(model(batched_yes[i])[2])[:,:,-1].detach().cpu()
    n = t.stack(model(batched_no[i])[2])[:,:,-1].detach().cpu()
    acts = t.cat((acts, t.stack((y,n), dim=1)), dim=2) 
    
# acts = t.cat([t.stack([t.stack(model(batched_yes[b].to('cuda'))[2])[:,:,-1].cpu().detach(), t.stack(model(batched_no[b].to('cuda'))[2])[:,:,-1].cpu().detach()]) for b in tqdm(range(len(batched_yes)))], dim=2)
acts = rearrange(acts, 'n y x a -> x y n a')
print(acts.shape)

# cache activations somewhere
t.save(acts, f'activations/{model_name}/activations.pt')
# %%
