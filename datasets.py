import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, BertTokenizerFast, GPT2TokenizerFast
import numpy as np
import random
import os
from ast import literal_eval

import json
import pandas as pd
import numpy as np

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")
BERT_PATH = os.path.join("..","BERT", "bert-base-uncased")
GPT2_PATH = os.path.join("..", "GPT2")

BERT_START_INDEX = 101
BERT_END_INDEX = 102

nyt_topics = ["tennis",
"housing",
"civil war and guerrilla warfare",
"golf",
"deaths (obituaries)",
"dancing",
"women",
"blacks",
"mergers, acquisitions and divestitures",
"hijacking",
"police",
"news and news media",
"stocks and bonds",
"research",
"election issues",
"world trade center (nyc)",
"accidents and safety",
"budgets and budgeting",
"labor",
"hockey, ice",
"children and youth",
"education and schools",
"suits and litigation",
"united states politics and government",
"computers and the internet",
"economic conditions and trends",
"murders and attempted murders",
"playoff games",
"art",
"law and legislation",
"restaurants",
"medicine and health",
"airlines and airplanes",
"books and literature",
"ethics",
"motion pictures",
"theater",
"united states armament and defense",
"finances",
"music",
"elections",
"television",
"terrorism",
"united states international relations",
"basketball",
"politics and government",
"football",
"biographical information",
"baseball",
"reviews"
]

# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\nytg\\corpus.json"
data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\arxiv_cornell\\arxiv_cornell.csv"

class PapersDataset(Dataset):

    def __init__(self, data_dir, encoder, axis, time_precision, train, max_len = 512, seed = 1):
        super(PapersDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        self.max_len = max_len
        self.encoder = encoder
        self.axis = axis
        self.time_precision = time_precision

        if encoder == "DistilBERT":
          self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
          self.tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
        elif encoder == "GPT2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_PATH)

        self.data = pd.read_csv(data_dir, encoding = 'utf-8', sep=';',
                                dtype={'id':str},
                                converters={'authors': literal_eval, 'topics':literal_eval})
        self.data = self.data.explode(self.axis)

        # self.data['date'] = pd.to_datetime(self.data.update_date, format = '%d/%m/%Y')
        self.data['date'] = pd.to_datetime(self.data['created'], format = "%a, %d %b %Y", exact=False)
        self.data = self.data[self.data['date']>'01/01/2010']

        self.min_date = min(self.data.date)
        self.max_date = max(self.data.date)

        if self.train:
            self.data, _ = train_test_split(
                                            self.data,
                                            test_size=0.3,
                                            stratify=self.data[[self.axis]],
                                            random_state=self.seed
                                            )
        else:
            _, self.data = train_test_split(
                                            self.data,
                                            test_size=0.3,
                                            stratify=self.data[[self.axis]],
                                            random_state=self.seed
                                            )

        self.data = self.data.sort_values([self.axis, 'date'])

        self.axis2id = {a:i for i, a in enumerate(self.data[self.axis].unique())}
        self.id2axis = {i:a for a,i in self.axis2id.items()}

        self.data['ddelta'] = (self.data.date - self.min_date).dt.days

        self.data['pdelta'] = (self.data.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

        self.data['start_pin'] = self.data.groupby(self.axis)['ddelta'].transform('min')
        self.data['end_pin'] = self.data.groupby(self.axis)['ddelta'].transform('max')

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")

        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)

        self.data.columns = ['texts' if x=='abstract' else x for x in self.data.columns]

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'title', 'texts',
                                        'pdelta', 'ddelta', 'start_pin', 'end_pin']].to_dict('records')

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        item = self.processed_data[index]
        doc_num = item['doc_id']

        if doc_num == 0:
            index+=2
        if doc_num == 1:
            index+=1

        item = self.processed_data[index]
        doc_num = item['doc_id']

        T = doc_num

        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]
        y_t = self.processed_data[index - T + t2]
        y_T = self.processed_data[index]

        t_ = t1
        t = t2

        total_docs = item['corpus_length']
        result = {
            'y_0': y_0['texts'],
            'y_t': y_t['texts'],
            'y_T': y_T['texts'],
            't_': y_0['ddelta'],
            't': y_t['ddelta'],
            'T': y_T['ddelta'],
            'total_t': total_docs,
            'class_t_': y_0['pdelta'],
            'class_t': y_t['pdelta'],
            'class_T': y_T['pdelta'],
            'axis':y_0['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)


class NytDataset(Dataset):

    def __init__(self, data_dir, encoder, time_precision, train, axis='authors', max_len = 512, seed = 1):
        super(NytDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        self.max_len = max_len
        self.encoder = encoder
        self.axis = axis
        self.time_precision = time_precision

        with open(self.data_dir, 'r') as f:
            corpus = f.readlines()

        self.data = []
        for c in corpus:
            self.data.append(json.loads(c))

        self.data = pd.DataFrame(self.data)
        self.data = self.data.explode('authors').explode('texts')

        self.data = self.data.explode(self.axis)

        if self.axis == 'topics':
            self.data = self.data[self.data.topics.isin(nyt_topics)]

        self.data['date'] = pd.to_datetime(self.data.date, format='%Y-%m-%d')

        self.min_date = min(self.data.date)
        self.max_date = max(self.data.date)

        if encoder == "DistilBERT":
          self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
          self.tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
        elif encoder == "GPT2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_PATH)

        if self.train:
            self.data, _ = train_test_split(
                                            self.data,
                                            test_size=0.2,
                                            stratify=self.data[[self.axis]],
                                            random_state=self.seed
                                            )
        else:
            _, self.data = train_test_split(
                                            self.data,
                                            test_size=0.2,
                                            stratify=self.data[[self.axis]],
                                            random_state=self.seed
                                            )

        self.data = self.data.sort_values([self.axis, 'date'])

        self.axis2id = {a:i for i, a in enumerate(self.data[self.axis].unique())}
        self.id2axis = {i:a for a,i in self.axis2id.items()}

        self.data['ddelta'] = (self.data.date - self.min_date).dt.days

        self.data['pdelta'] = (self.data.date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

        self.data['start_pin'] = self.data.groupby(self.axis)['ddelta'].transform('min')
        self.data['end_pin'] = self.data.groupby(self.axis)['ddelta'].transform('max')

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")
        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'pdelta', 'ddelta', 'start_pin', 'end_pin']].to_dict('records')

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, truncation=True, max_length = self.max_len, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        item = self.processed_data[index]
        doc_num = item['doc_id']

        if doc_num == 0:
            index+=2
        if doc_num == 1:
            index+=1

        item = self.processed_data[index]
        doc_num = item['doc_id']

        T = doc_num

        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]
        y_t = self.processed_data[index - T + t2]
        y_T = self.processed_data[index]

        t_ = t1
        t = t2

        total_docs = item['corpus_length']
        result = {
            'y_0': y_0['texts'],
            'y_t': y_t['texts'],
            'y_T': y_T['texts'],
            't_': y_0['ddelta'],
            't': y_t['ddelta'],
            'T': y_T['ddelta'],
            'total_t': total_docs,
            'class_t_': y_0['pdelta'],
            'class_t': y_t['pdelta'],
            'class_T': y_T['pdelta'],
            'axis':y_0['axis_id'],
            'start_pin':item['start_pin'],
            'end_pin':item['end_pin']
        }

        return result

    def __len__(self):
        return len(self.processed_data)
