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

# def set_seed(graine):
#     random.seed(graine)
#     np.random.seed(graine)
#     torch.manual_seed(graine)
#     torch.cuda.manual_seed_all(graine)

############# Text Reader ###############
# def clean_str(string):
#     string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip()
    
# def read(file_path):
#     with open(file_path, mode='r', encoding='utf-8') as f_in:
#         content=clean_str(f_in.read())
#     return(content)

# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\nytg\\imputation\\corpus.json"
# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\arxiv_cornell\\arxiv_40.csv"

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

        self.data['update_date'] = pd.to_datetime(self.data.update_date, format = '%d/%m/%Y')

        self.min_date = min(self.data.update_date)
        self.max_date = max(self.data.update_date)

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

        self.data = self.data.sort_values([self.axis, 'update_date'])

        self.axis2id = {a:i for i, a in enumerate(self.data[self.axis].unique())}
        self.id2axis = {i:a for a,i in self.axis2id.items()}

        self.data['ddelta'] = (self.data.update_date - self.min_date).dt.days

        self.data['pdelta'] = (self.data.update_date.dt.to_period(self.time_precision).view('int64') - self.min_date.to_period(self.time_precision).ordinal)

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")
        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'title', 'abstract', 'pdelta', 'ddelta']].to_dict('records')

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, return_tensors='pt')

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
            'y_0': y_0['abstract'],
            'y_t': y_t['abstract'],
            'y_T': y_T['abstract'],
            't_': y_0['ddelta'],
            't': y_t['ddelta'],
            'T': y_T['ddelta'],
            'total_t': total_docs,
            'class_t_': y_0['pdelta'],
            'class_t': y_t['pdelta'],
            'class_T': y_T['pdelta'],
            'axis':y_0['axis_id']
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

        self.data['corpus_length'] = self.data.groupby(self.axis)[self.axis].transform("count")
        self.data['doc_id'] = self.data.groupby(self.axis).cumcount()
        self.data['axis_id'] = self.data[self.axis].map(self.axis2id)

        self.processed_data = self.data[[self.axis, 'axis_id', 'corpus_length', 'doc_id', 'texts', 'pdelta', 'ddelta']].to_dict('records')

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
            'axis':y_0['axis_id']
        }

        return result

    def __len__(self):
        return len(self.processed_data)

# arxiv = "C:\\Users\\EnzoT\\Documents\\datasets\\arxiv\\arxiv.json"

# with open(arxiv, 'r') as f:
#     data = json.load(f)

# for d in data:
#     d['author'] = [v['name'] for v in eval(d['author'])]
#     d['link'] = [v['href'] for v in eval(d['link'])]
#     d['tag'] = [v['term'] for v in eval(d['tag']) if v['term'] in taxonomy]

# df = pd.DataFrame.from_dict(data)
# df = df.explode("author")
# df['count'] = df.groupby("author")["author"].transform("count")

# df[["author", "count"]].drop_duplicates("author")['count'].describe()

# df2 = df[df["count"]>19]

# print(df2[["author", "count"]].drop_duplicates("author")['count'].describe())

# print(len(df2))
# print(len(df2.author.unique()))

# df2.head()

# s2g = "C:\\Users\\EnzoT\\Documents\\datasets\\s2g\\modeling\\corpus.json"

# with open(s2g, 'r') as f:
#     data = f.readlines()

# nytg = "C:\\Users\\EnzoT\\Documents\\datasets\\nytg\\modeling\\corpus.json"

# with open(nytg, 'r') as f:
#     data = f.readlines()

# data = [json.loads(a) for a in data]

# df = pd.DataFrame.from_dict(data)

# df = df.explode("authors")
# df = df.explode("texts")

# df['count'] = df.groupby("authors")["authors"].transform("count")

# df[["authors", "count"]].drop_duplicates("authors")['count'].describe()



# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\arxiv_cornell\\arxiv_cornell.json"

# with open(data_dir, 'r', encoding='utf-8') as f:
#     data = f.readlines()

# papers = []
# for paper in data:
#     papers.append(json.loads(paper))

# for paper in papers:
#     paper.update(paper["versions"][-1])

# df = pd.DataFrame.from_dict(papers)

# df['update_date'] = pd.to_datetime(df["update_date"])

# # Max numbers of versions : 187
# # Average number of versions : 1.57

# start = datetime(2010,1,1)
# end = datetime(2020,1,1)

# mask = (df["update_date"] > start) & (df["update_date"] < end)

# df = df.loc[mask]

# df = df[df.categories.str.contains("|".join(cs_cat))]

# df = df.explode('authors_parsed')
# df['authors_parsed'] = df['authors_parsed'].str.join(' ').str.strip()
# df['count'] = df.groupby("authors_parsed")["authors_parsed"].transform("count")

# df2 = df[df["count"]>39]

# print(df2[["authors_parsed", "count"]].drop_duplicates("authors_parsed")['count'].describe())


# print(len(df2))
# print(len(df2.author.unique()))

# df2.head()


# authors = list(df["authors_parsed"])
# authors = [[' '.join(a).strip() for a in author] for author in authors]
# authors = [item for sublist in authors for item in sublist]

# authors_count = Counter(authors)

# authors_count = pd.DataFrame(dict(authors_count).items(), columns=["author", "number"])
# authors_count = authors_count.sort_values('number', ascending=False)

# authors_count[authors_count.number >= 100]

# taxonomy = ['acc-phys', 'adap-org', 'alg-geom', 'ao-sci', 'astro-ph',
#  'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR', 'atom-ph',
#  'bayes-an', 'chao-dyn', 'chem-ph', 'cmp-lg', 'comp-gas', 'cond-mat', 'cond-mat.dis-nn', 'cond-mat.mes-hall',
#  'cond-mat.mtrl-sci', 'cond-mat.other', 'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech',
#  'cond-mat.str-el', 'cond-mat.supr-con', 'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
#  'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC',
#  'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
#  'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY', 'dg-ga', 'econ.EM', 'econ.GN',
#  'econ.TH', 'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY', 'funct-an', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph',
#  'hep-th', 'math-ph', 'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO', 'math.CT',
#  'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT', 'math.HO',
#   'math.IT', 'math.KT', 'math.LO', 'math.MG', 'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC', 
#  'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG', 'math.SP', 'math.ST', 'mtrl-th', 'nlin.AO',
#  'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI', 'nucl-ex', 'nucl-th', 'patt-sol', 'physics.acc-ph', 'physics.ao-ph',
#  'physics.app-ph', 'physics.atm-clus', 'physics.atom-ph', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph',
#  'physics.comp-ph', 'physics.data-an', 'physics.ed-ph', 'physics.flu-dyn', 'physics.gen-ph', 'physics.geo-ph',
#  'physics.hist-ph', 'physics.ins-det', 'physics.med-ph', 'physics.optics', 'physics.plasm-ph', 'physics.pop-ph',
#  'physics.soc-ph', 'physics.space-ph', 'plasm-ph', 'q-alg', 'q-bio', 'q-bio.BM', 'q-bio.CB', 'q-bio.GN',
#  'q-bio.MN', 'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO', 'q-fin.CP', 
#  'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR', 
#  'quant-ph', 'solv-int', 'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT', 'stat.TH', 'supr-con',
#  'test','test.dis-nn', 'test.mes-hall','test.mtrl-sci','test.soft','test.stat-mech',
#  'test.str-el','test.supr-con']

# cs_cat = ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
#  'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC',
#  'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
#  'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY']