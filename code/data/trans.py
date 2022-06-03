import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm_notebook as tqdm
import os
import re
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

# Load translation model
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

en2ru.cuda()
ru2en.cuda()

en2de.cuda()
de2en.cuda()

path = './'
train_df = pd.read_csv(path+'train.csv', header=None)
train_df.head()

train_labels = [v-1 for v in train_df[0]]
train_text = [v for v in train_df[2]]

train_text[0]

len(train_text)

max(train_labels)

# split and get our unlabeled training data
def train_val_split(labels, n_labeled_per_class, n_labels, seed = 0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[:])
        val_idxs.extend(idxs[-3000:])
    
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_labels, 500, 4)

len(train_unlabeled_idxs)

idxs = train_unlabeled_idxs

idxs[0]

# back translate using Russian as middle language
def translate_ru(start, end, file_name):
    trans_result = {}
    for id in tqdm(range(start, end)):
        trans_result[idxs[id]] = ru2en.translate(en2ru.translate(train_text[idxs[id]],  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)

# back translate using German as middle language
def translate_de(start, end, file_name):
    trans_result = {}
    for id in tqdm(range(start, end)):
        trans_result[idxs[id]] = de2en.translate(en2de.translate(train_text[idxs[id]],  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)

translate_de(0,120000, 'de_1.pkl')

translate_ru(0,120000, 'ru_1.pkl')