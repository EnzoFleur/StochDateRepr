import torch
import torch.nn as nn
from transformers import DistilBertModel, GPT2Model, BertModel
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_PATH = os.path.join("..", "BERT", "bert-base-uncased")
GPT2_PATH = os.path.join("..", "GPT2")

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False

class DAN(nn.Module):

    def __init__(self, input_dim, hidden, r):
        super(DAN, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.r = r

        self.do1 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.do2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, r)

    def forward(self, x):

        x = x.mean(dim=1)
        x = self.do1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.do2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x

class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

            self.mlp = nn.Sequential(*[
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            ])
        
        def forward(self, x):
            return self.mlp(x)

class BrownianEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, loss, H,
                tokenizer,
                finetune, method):
        super(BrownianEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune
        self.tokenizer = tokenizer
        self.loss = loss

        self.method = method 

        print(self.method)

        if self.tokenizer == "DistilBERT":
          self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        elif self.tokenizer == "BERT":
          self.encoder = BertModel.from_pretrained(BERT_PATH)
        elif self.tokenizer == "GPT2":
          self.encoder = GPT2Model.from_pretrained(GPT2_PATH)
        
        for param in self.encoder.parameters():
            param.requires_grad = self.finetune

        self.mlp = MLP(768, self.hidden_dim, self.latent_dim)

        self.C_eta = nn.Linear(1, 1)

        self.params = nn.ModuleDict({
            'encoder': nn.ModuleList([self.encoder]),
            'classifier': nn.ModuleList([self.mlp, self.C_eta])
        })
        
        # Switch off bias in linear layers
        # self.mlp.apply(weights_init)
        # self.log_q.apply(weights_init)
        # self.C_eta.apply(weights_init)


    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def forward(self, input_ids, attention_mask):

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, attention_mask)

        latent_state = self.mlp(hidden_state)

        return latent_state

    def encode_doc(self, input_ids, attention_mask):
        
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, attention_mask)

        return hidden_state