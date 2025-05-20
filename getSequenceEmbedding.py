import torch
import h5py
import pandas as pd
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from tqdm import tqdm
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
import math
import h5py
from tqdm import tqdm


def Seq_to_vec(Sequence):  # , model, tokenizer, device
    print('Embedding ...')
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    print("the length of sequence list:", len(Sequence))
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    #
    # features = []
    # for i in tqdm(range(len(sequences_Example)), total=len(sequences_Example), desc='Processing', unit='row'):
    #     # print('For sequence ', str(i + 1))
    #     sequences_Example_i = sequences_Example[i]
    #     sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
    #     ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
    #     input_ids = torch.tensor(ids['input_ids']).to(device)
    #     attention_mask = torch.tensor(ids['attention_mask']).to(device)
    #     with torch.no_grad():
    #         embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    #     embedding = embedding.last_hidden_state.cpu().numpy()
    #     for seq_num in range(len(embedding)):
    #         seq_len = (attention_mask[seq_num] == 1).sum()
    #         seq_emd = embedding[seq_num][:seq_len - 1]
    #         features.append(seq_emd)
    tokenizer = T5Tokenizer.from_pretrained("../processing/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("../processing/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in tqdm(range(len(sequences_Example)), total=len(sequences_Example), desc='Processing', unit='row'):
        # for i in range(len(sequences_Example)):
        #     print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            seq_emd = np.mean(seq_emd, axis=0)
            # print(seq_emd.shape)
            features.append(seq_emd)
    return features


def save_Seq_to_vec(sequences, save_path):
    ###### you should place downloaded model into this directory.
    # print('Loading Model Parameters...')
    # tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", W do_lower_case=False)
    # model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    # gc.collect()
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # gpu memory is not enough cuda:0
    # print(f"Using {device}...")
    # model = model.to(device)
    # model = model.eval()
    # sequence_input_3d = Seq_to_vec(sequences, model, tokenizer, device)  # .values
    sequence_input_3d = Seq_to_vec(sequences)  # .values

    n = len(sequences)
    print('Saving Data into HDF5 file...')
    with h5py.File(save_path, 'w') as f:  #
        for i in tqdm(range(n), total=n, unit='row', desc="Processing"):
            f.create_dataset(str(i), data=sequence_input_3d[i])
        # print(f"Sequence {l} to {r} have been saved...{r / 17010.0 * 100}%")
    print("Save Completed...")


def convertToXdim(load_path, save_path, dim=128):
    with h5py.File(save_path, 'w') as fw:
        with h5py.File(load_path, 'r') as fr:
            n = len(fr)
            for i in tqdm(range(n), total=n, desc='Processing', unit='row'):
                i_name = str(i)
                mat = fr[i_name][()]
                lines_cnt = len(mat)
                mat = mat.reshape(lines_cnt, dim, -1)
                mat = np.mean(mat, axis=2)

                fw.create_dataset(i_name, data=mat)


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm) > 218:
            # print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109] + sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def save_Smi_to_vec(smiles, save_path):
    smiles_input_3d = smiles_to_vec(smiles)  # .values
    print(smiles_input_3d.shape)
    n = len(smiles)
    print('Saving Data into HDF5 file...')
    with h5py.File(save_path, 'w') as f:  #
        for i in tqdm(range(n), total=n, unit='row', desc="Processing"):
            f.create_dataset(str(i), data=smiles_input_3d[i])
        # print(f"Sequence {l} to {r} have been saved...{r / 17010.0 * 100}%")
    print("Save Completed...")


if __name__ == '__main__':
    # df = pd.read_csv('../../data/brenda_sabiork_turnover_number_all_keepMax.csv')
    # df = pd.read_csv('../../data/DLTKcat_Me_training_final.csv')
    df = pd.read_csv('../../data/DLTKcat_Me_testing_final2.csv')
    sequences = df['seq']
    print(len(sequences))
    # save_path = '../../data/EnzymeOfKcat_29681entries_1024dim_1000maxlen_mean.h5'
    # save_path = '../../data/DLTKcat_Me_training_seq.h5'
    save_path = '../../data/DLTKcat_Me_testing_seq.h5'
    save_Seq_to_vec(sequences, save_path)

    smiles = df['smiles']
    # save_path = '../../data/EnzymeOfKcat_29681entries_smiles_mean.h5'
    # save_path = '../../data/DLTKcat_Me_training_smi.h5'
    save_path = '../../data/DLTKcat_Me_testing_smi.h5'
    save_Smi_to_vec(smiles, save_path)

    # load_path = save_path
    # save_path = '../../data/EnzymeOfKcat_16838entries_128dim_1000maxlen.h5'
    # convertToXdim(load_path, save_path)

    # with h5py.File("../../data/SmilesOfKcat_16838entries_128dim_113maxlen.h5", 'r') as f:
    #     n = len(f)
    #     maxx = 0
    #     for i in range(n):
    #         i_name = str(i)
    #         maxx = max(maxx, len(f[i_name][()]))
    #         print(i)
    #     print(maxx)