import pandas as pd
from tqdm import tqdm
import numpy as np
from random import seed
import os
import argparse
from datetime import datetime

from sklearn.preprocessing import normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score, mean_absolute_error, accuracy_score

from encoders import DISTILBERT_PATH, BrownianEncoder
from brownianlosses import BrownianBridgeLoss, BrownianLoss

from transformers import get_linear_schedule_with_warmup    
from datasets import PapersDataset, NytDataset

# Setting up the device for GPU usage
from torch import cuda

import torch
from torch.utils.data import DataLoader

import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend = 'nccl',
                        init_method = 'env://',
                        world_size=idr_torch.size,
                        rank = idr_torch.rank)

torch.cuda.set_device(idr_torch.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\arxiv_cornell\\arxiv_cornell.csv"
# data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\nytg\\corpus.json"

# BATCH_SIZE = 8
# REDUCED_BS = BATCH_SIZE
# EPOCHS = 10
# LEARNING_RATE = 1e-4
# LATENT_SIZE = 32
# ENCODER = 'DistilBERT'
# FINETUNE = False
# LOSS = "fBM"
# HURST = 0.9
# TIME = 'Y'
# AXIS = 'authors'
# PINNING = True

if __name__ == "__main__":

    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-bs','--batchsize', default=32, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no-finetune', dest='finetune', action='store_false')
    parser.set_defaults(finetune=False)
    parser.add_argument('-lr','--learningrate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('-l','--loss', default="BB", type=str,
                        help='Loss (either BB or fBM')
    parser.add_argument('-e','--encoder', default="DistilBERT", type=str,
                        help='Language encoder')
    parser.add_argument('-hu','--hurst', default=1/2, type=float,
                        help='Hurst parameter (if loss is BB)')
    parser.add_argument('-es','--embeddingsize', default=32, type=int,
                        help='Size of the latent representation')
    parser.add_argument('--pinning', action='store_true')
    parser.add_argument('--no-pinning', dest='pinning', action='store_false')
    parser.set_defaults(pinning=False)
    parser.add_argument('-ax','--axis', default='authors', type=str,
                        help='Axis defining trajectories (either authors or topics)')
    parser.add_argument('-t','--timeprecision', default='Y', type=str,
                        help='Time precision for date classification (Y, M or D)')
    args = parser.parse_args()

    data_dir = args.dataset
    DATASET = data_dir.split(os.sep)[-2]
    BATCH_SIZE = args.batchsize
    REDUCED_BS = BATCH_SIZE // idr_torch.size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learningrate
    ENCODER = args.encoder
    LOSS = args.loss
    HURST = args.hurst
    FINETUNE  = args.finetune
    PINNING = args.pinning
    LATENT_SIZE = args.embeddingsize
    AXIS = args.axis
    TIME = args.timeprecision

    if DATASET == "arxiv_cornell":
        data_dir = data_dir.replace("cornell.csv", AXIS + ".csv")
        dataset_train = PapersDataset(data_dir = data_dir, encoder=ENCODER, train=True, seed=42, axis=AXIS, time_precision=TIME)
        dataset_test = PapersDataset(data_dir = data_dir, encoder=ENCODER, train=False, seed=42, axis=AXIS, time_precision=TIME)
    elif DATASET == "nytg":
        dataset_train = NytDataset(data_dir = data_dir, encoder=ENCODER, train=True, seed=42, time_precision=TIME)
        dataset_test = NytDataset(data_dir = data_dir, encoder=ENCODER, train=False, seed=42, time_precision=TIME)
    
    method = "%s_FT%d_P%d_%s_H%0.2f" % (ENCODER, FINETUNE, PINNING, LOSS, HURST)

    model = BrownianEncoder(hidden_dim=512, latent_dim=LATENT_SIZE,
                            loss = LOSS,
                            H=HURST,
                            tokenizer = ENCODER,
                            finetune = FINETUNE, method = method).to(device)

    ddp_model = DDP(model, device_ids=[idr_torch.local_rank], find_unused_parameters=True)

    total_steps = len(dataset_train) * EPOCHS

    # optimizer = torch.optim.Adam(params = [
    #         {'params':ddp_model.module.encoder.parameters(), 'lr':3e-4},
    #         {'params':ddp_model.module.classifier.parameters(), 'lr':LEARNING_RATE}
    #     ])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = LEARNING_RATE)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

    def get_loss_batch(batch, model, loss):

        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']

        t_s = torch.Tensor(batch['t_'].float()).to(device)
        ts = torch.Tensor(batch['t'].float()).to(device)
        Ts = torch.Tensor(batch['T'].float()).to(device)

        input_ids, attention_masks = dataset_train.tokenize_caption(obs_0, device)
        z_0 = model(input_ids, attention_masks)

        input_ids, attention_masks = dataset_train.tokenize_caption(obs_t, device)
        z_t = model(input_ids, attention_masks)

        input_ids, attention_masks = dataset_train.tokenize_caption(obs_T, device)
        z_T = model(input_ids, attention_masks)

        # log_q_y_T = model.get_log_q(z_t)

        if loss == "BB":
            loss_fn = BrownianBridgeLoss(
                        z_0=z_0,
                        z_t=z_t,
                        z_T=z_T,
                        t_=t_s,
                        t=ts,
                        T=Ts,
                        pin=PINNING, start_pin=batch['start_pin'], end_pin=batch['end_pin'],
                        # log_q_y_T=log_q_y_T,
                        max_seq_len=torch.Tensor(batch['total_t'].float()).to(device),
                        H=HURST,
                        eps=1e-4,
                        label=batch['axis']
                    )
        elif loss == "fBM":
            loss_fn = BrownianLoss(
                z_0=z_0,
                z_t=z_t,
                z_T=z_T,
                t_=t_s,
                t=ts,
                T=Ts,
                pin=PINNING, start_pin=batch['start_pin'], end_pin=batch['end_pin'],
                # log_q_y_T=log_q_y_T,
                max_seq_len=torch.Tensor(batch['total_t'].float()).to(device),
                H=HURST,
                eps=1e-4,
                label=batch['axis']
            )

        return loss_fn.get_loss()

    def cosine_similarity(aut_embeddings, time_embeddings, doc_embeddings, axis, times):

        aut_embeddings = normalize(aut_embeddings, axis=1)
        doc_embeddings = normalize(np.vstack(doc_embeddings), axis=1)

        nd = len(doc_embeddings)
        na = len(aut_embeddings)

        aut_doc_test = np.zeros((nd, na))
        aut_doc_test[[i for i in range(nd)], [ax for ax in axis]] = 1

        y_score = normalize( doc_embeddings @ aut_embeddings.transpose(),norm="l1")
        ce = coverage_error(aut_doc_test, y_score)/na*100
        lr = label_ranking_average_precision_score(aut_doc_test, y_score)*100

        time_embeddings = normalize(time_embeddings, axis=1)

        y_score = normalize(doc_embeddings @ time_embeddings.transpose(),norm="l1")
        y_pred = np.argmax(y_score, axis=1)

        mae = mean_absolute_error(times, y_pred)
        acc = accuracy_score(times, y_pred)

        return ce, lr, mae, acc

    def eval(model, dataset_test, dataset_train):
        corpus_lengths = list(dataset_train.data[['axis_id', 'corpus_length']].drop_duplicates('axis_id')['corpus_length'])
        texts = list(dataset_train.data.texts)

        author_embeddings = []
        time_embeddings = []
        doc_embeddings = []

        i=0
        for l in corpus_lengths:
            corpus = texts[i:i+l]
            author_embedding = []

            for c in chunks(corpus, REDUCED_BS):

                input_ids, attention_masks = dataset_train.tokenize_caption(c, device)
                z_0 = model(input_ids, attention_masks).cpu().numpy().mean(axis=0)

                author_embedding.append(z_0)

            author_embeddings.append(np.mean(author_embedding,axis=0))
            i=+l

        for i, row in dataset_train.data[['timestep', 'texts']].sort_values('timestep').groupby('timestep'):
            corpus = list(row['texts'])

            time_embedding = []

            for c in chunks(corpus, REDUCED_BS):

                input_ids, attention_masks = dataset_train.tokenize_caption(c, device)
                z_0 = model(input_ids, attention_masks).cpu().numpy().mean(axis=0)

                time_embedding.append(z_0)

            time_embeddings.append(np.mean(author_embedding,axis=0))

        time_embeddings = np.vstack(time_embeddings)

        for c in chunks(list(dataset_test.data.texts), REDUCED_BS):

            input_ids, attention_masks = dataset_train.tokenize_caption(c, device)
            z_0 = model(input_ids, attention_masks).cpu().numpy()

            doc_embeddings.append(z_0)

        doc_embeddings = np.vstack(doc_embeddings)

        np.save(os.path.join("model", DATASET, AXIS, "%s_docs.npy" % (model.module.method)), doc_embeddings)

        author_embeddings = np.vstack(author_embeddings)

        ce, lr, mae, acc = cosine_similarity(author_embeddings, time_embeddings, doc_embeddings, dataset_test.data.axis_id, dataset_test.data.timestep)
        
        return ce, lr, mae, acc

    def fit(epochs, model, optimizer, scheduler, dataset_train, dataset_test):

        dataloader_train = DataLoader(dataset_train, batch_size=REDUCED_BS, shuffle=True, pin_memory=True)
        dataloader_test = DataLoader(dataset_test, batch_size=REDUCED_BS, shuffle=False, pin_memory=True)

        loss_eval = 0
        for epoch in range(1, epochs+1):

            if idr_torch.rank == 0: start = datetime.now()

            model.train()

            loss_training = 0
            for batch in tqdm(dataloader_train):  

                loss = get_loss_batch(batch, model, model.module.loss)
                
                optimizer.zero_grad()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
                optimizer.step()

                scheduler.step()

                loss_training+= loss.item()

            loss_training /= len(dataloader_train)

            if (epoch % 2 == 0):

                if (idr_torch.rank == 0):
                    model.eval()
                    
                    if not os.path.isdir(os.path.join("model", DATASET, AXIS)):
                        os.mkdir(os.path.join("model", DATASET, AXIS))

                    torch.save(model, os.path.join("model", DATASET, AXIS, "%s_ckpt.pt" % (model.module.method)))

                    with torch.no_grad():
                        loss_eval = 0
                        for batch in tqdm(dataloader_test):

                            loss = get_loss_batch(batch, model, model.module.loss)
                            loss_eval+= loss.item()

                        loss_eval/=len(dataloader_test)

                        ce, lr, mae, acc = eval(model, dataset_test, dataset_train)               

            if (idr_torch.rank == 0):
                print("[%d/%d] in %s Evaluation loss : %.4f  |  Training loss : %.4f \n" % (epoch, epochs, str(datetime.now() - start), loss_eval, loss_training), flush=True)
                if not model.module.training:
                    print("Coverage : %.2f  | Precision : %.2f | MAE : %.1f  | Accuracy : %.2f \n" % (ce, lr, mae, acc), flush=True)

    fit(EPOCHS, ddp_model, optimizer, scheduler, dataset_train, dataset_test)

    if (idr_torch.rank == 0):
        print("We're finished !")
