from einops import rearrange
from tqdm import tqdm
import torch as t

def get_activations(model, tokenizer, dataset, layers, frac): 
    """
    Takes in modiified TQA dictionary with 'yes', 'no', and 'label' keys 
    and outptus activations for each layer in layers. Output shape is 
    (len(dataset), 2, len(layers)*len(activation_at_layer))
    """
    dlen = int(frac*len(dataset))
    activations = t.stack([t.stack([t.stack([get_one_activation(model, tokenizer, dataset[i][j], [layer]) for layer in layers]) for j in ['yes', 'no']]) for i in tqdm(range(dlen))])
    activations = rearrange(activations, 'i j k l -> i j (k l)') 
    labels = t.tensor([dataset[i]['label'] for i in range(dlen)])
    return activations, labels
    

def get_one_activation(model, tokenizer, prompt, layers): 
    """
    Outputs activations at hidden_layer layer of model on the last token of prompt
    """
    input = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return rearrange(t.stack([model(input)[2][layer][0,-1] for layer in layers]), 'i j -> (i j)')

def get_normalized_score(model, tokenizer, dataset, labels): 
    """
    Takes in modiified TQA dictionary with 'yes', 'no', and 'label' keys 
    and iterates through the questions to find the likelihood of the answer tokens
    """
    dlen = len(dataset)
    correct_prob = []
    for i in tqdm(range(dlen)): 
        question = dataset[i]['yes'][:-3]
        yes_logit = get_logit(model, tokenizer, question, 'Yes')
        no_logit = get_logit(model, tokenizer, question, 'No')
        p = t.softmax(t.stack([yes_logit, no_logit]), dim=0) 
        correct_prob.append(p[labels[i]])
    return sum(correct_prob)/len(correct_prob)

def get_logit(model, tokenizer, prompt, answer):
    """
    Outputs logit of answer token in prompt
    """
    input = tokenizer(prompt, return_tensors="pt")["input_ids"]
    logits = model(input)[0][0,-1,:].detach()
    return logits[tokenizer(answer).input_ids[0]]