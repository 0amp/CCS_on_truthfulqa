
# %%
%load_ext autoreload
%autoreload 2

# %%
import torch as t
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from CCS import CCS
from GPT2Wrapper import GPT2Wrapper
from utils import * 
import seaborn as sns
import matplotlib.pyplot as plt
# %%
# load activations and labels
all_acts = t.load('activations/gpt2-xl/activations.pt')
print(all_acts.shape)
activations = all_acts[:,:,24,:]
print(activations.shape)

labels = t.load('data/labels.pt')
# %%
# load model
model = GPT2Wrapper('gpt2-xl', device='cpu')

# test model generate
prompt = 'The sky is blue. The grass is green. The sun is yellow. The moon is'
print(model.generate(prompt))

# %%
tn = 500
x0 = activations[:tn,0,:]
x0 = (x0 - x0.mean(axis=0, keepdims=True))/x0.std(axis=0, keepdims=True)
x1 = activations[:tn,1,:]
x1 = (x1 - x1.mean(axis=0, keepdims=True))/x1.std(axis=0, keepdims=True)
y = labels[:tn]
testx0 = activations[tn:,0,:]
testx0 = (testx0 - testx0.mean(axis=0, keepdims=True))/testx0.std(axis=0, keepdims=True)
testx1 = activations[tn:,1,:]
testx1 = (testx1 - testx1.mean(axis=0, keepdims=True))/testx1.std(axis=0, keepdims=True)
testy = labels[tn:]

model.train_CCS(x0, x1, y)

# WEIRD THINGS
## doesn't return actual max(acc, 1-acc) ! 
## add terms to loss maybe

# %%
print(model.CCS_pred_acc(testx0, testx1, testy))

# %%
# run CCS on tn data points for all potential layers and plot results
layer_accs = []
tn = 400
for i in range(5,all_acts.shape[2]-5): 
    acts = all_acts[:tn,:,i,:]
    x0, x1 = acts[:,0,:], acts[:,1,:]
    x0 = (x0 - x0.mean(axis=0, keepdims=True))/x0.std(axis=0, keepdims=True)
    x1 = (x1 - x1.mean(axis=0, keepdims=True))/x1.std(axis=0, keepdims=True)
    y, testy = labels[:tn], labels[tn:]
    xtest0, xtest1 = activations[tn:,0,:], activations[tn:,1,:]
    xtest0 = (xtest0 - xtest0.mean(axis=0, keepdims=True))/xtest0.std(axis=0, keepdims=True)
    xtest1 = (xtest1 - xtest1.mean(axis=0, keepdims=True))/xtest1.std(axis=0, keepdims=True)
    ccs = CCS(x0, x1, y, ntries=10)
    print("LAYER: ", i)
    ccs.train()
    pred = ccs.pred_acc(xtest0, xtest1, testy)
    pred = max(pred, 1-pred)
    print("PRED ACC: ", pred)
    layer_accs.append(pred)

# %%
x = list(range(5,all_acts.shape[2]-5))
zero_shot = 0.4888
# add dotted line for zero shot accuracy to sns plot with layer_accs
sns.lineplot(x = x, y = layer_accs)
plt.axhline(y=zero_shot, color='r', linestyle='--')
plt.show()

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
# RESULT2: 0.4688

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pandas as pd
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
config = GPT2Config.from_pretrained("gpt2-xl", output_hidden_states=True)
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", config=config) 

data = pd.read_csv('data/modifiedtqa.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'Yes' else 0)
data['yes'] = data['question'] + ' Yes'
data['no'] = data['question'] + ' No'
data = data.drop(['question'], axis=1)
data = data.to_dict('records')
data = data[300:450]

print(get_normalized_score(model, tokenizer, data, labels))
        
# %%
