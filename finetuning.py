# finetune GPT2-xl on CCS output 
# %%
% load_ext autoreload
% autoreload 2

# %%
import torch as t
import gc
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from CCS import CCS
from GPT2Wrapper import GPT2Wrapper
from utils import *
import seaborn as sns
# %%
# load data
gc.collect()
t.cuda.empty_cache()

# %% 
# set up model
model_name = 'gpt2-xl'
device = 'cuda'
gpt2 = GPT2Wrapper(model_name, device=device)

# %%
# test model
prompt = 'The quick brown fox jumps over the lazy dog'
gpt2.generate(prompt, num_tokens=100, temperature=1.0, top_k=0, top_p=0.9, repetition_penalty=1.0, do_sample=True, num_beams=1, early_stopping=False, use_cache=True)

# %%
# train CCS probe and check predictions
activations = t.load('activations/gpt2-xl/activations.pt')

# %%
# %%
