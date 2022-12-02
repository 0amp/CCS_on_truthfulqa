from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch as t
import torch.nn as nn
from CCS import CCS

class GPT2Wrapper(nn.Module):
    def __init__(self, model_name, device='cuda'):
        super().__init__()
        self.config = GPT2Config.from_pretrained(model_name, output_hidden_states=True)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.gpt2.to(device)

        self.device = device

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        return outputs
    
    def generate(self, prompt, num_tokens=100, temperature=1.0, top_k=0, top_p=0.9, repetition_penalty=1.0, do_sample=True, num_beams=1, early_stopping=False, use_cache=True, **model_kwargs):
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        return self.gpt2(input_ids, max_length=input_ids.shape[1] + num_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample, num_beams=num_beams, early_stopping=early_stopping, use_cache=use_cache, **model_kwargs)[0]
    
    def get_activations(self, prompt, layer=0, device='cuda'):
        self.gpt2.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        with t.no_grad():
            output = self.gpt2(input_ids)
            return output[2][layer].squeeze(0).cpu()
        
    def get_activations_batch(self, prompts, layer=0, device='cuda'):
        self.gpt2.eval()
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        with t.no_grad():
            output = self.gpt2(input_ids)
            return output[2][layer].cpu()
        
    def train_CCS(self, x0, x1, labels, epochs=100, lr=0.01, device='cuda', linear=True):
        self.ccs = CCS(x0, x1, labels, epochs=epochs, lr=lr, device=device, linear=linear)
        self.ccs.train()

    def get_CCS_acc(self, x0, x1, y): 
        return self.ccs.pred_acc(x0, x1, y)
    
    def get_CCS_pred(self, x): 
        return self.ccs.make_pred(x)

    def finetune(self, train_dataset, epochs=10, lr=1e-4, batch_size=8, device='cuda'):
        pass

    