

import os
import torch
import glob
import datetime
import numpy as np
import shutil
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from metric import *

def calculate_params(model, test_dataloader, device):

    model.eval()
    with torch.no_grad():

        errors = []

        for t in range(len(test_dataloader)):
#             data_x, data_y, indexes = data
#             data_x = data_x.to(device)

            _, _, error, _, _, _ = model(data_x, mode="test")
            errors.append(error)
            
        errors = torch.cat(errors, dim=0).view(-1, errors[0].size(-1)) #[seq_len, input_size]

        mean = errors.mean(0) #[input_size]
        cov = (errors.t()).mm(errors)/errors.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(1).t()) #[input_size, input_size]

        return mean, cov

def save_mean_cov(args, mean, cov):

    mean_cov_dict = {}
    mean_cov_dict['mean'] = mean
    mean_cov_dict['cov'] = cov
    with open('mean_cov.pkl', 'wb') as fw:
        pickle.dump(mean_cov_dict, fw)

def load_mean_cov(args):

    with open('mean_cov.pkl', 'rb') as fr:
        mean_cov_dict = pickle.load(fr)
        mean = mean_cov_dict['mean']
        cov = mean_cov_dict['cov']

    return mean, cov

def save_model_weights(args, model):

    torch.save(model.state_dict(), 'models/model_{}.pkl'.format(args.window_length))

def load_model_weights(args, model):

    model.load_state_dict(torch.load('models/model_{}.pkl'.format(args.window_length)))


def test(args, model, dataloader, target="test"):

    print("Testing the Model!!!")
    outputs, score, inputs, labels, errors, all_indexes =  get_anomaly_socre(args, model, dataloader, target)

    precision_01, recall_01, f_01 = get_precison_recall(score.cpu(), labels.cpu(), 10000, beta=0.1)
    print("F01_score: ", f_01.cpu().data.numpy())

    precision, recall, f_1 = get_precison_recall(score.cpu(), labels.cpu(), 10000, beta=1.0)
    print("F1_score: ", f_1.cpu().data.numpy())

    _, _, roc_auc = CalculateROCAUCMetrics(score, labels)
    _, _, pr_auc = CalculatePrecisionRecallCurve(score, labels)
    print("ROC-AUC:{}, PR-AUC:{}".format(roc_auc, pr_auc))
    
    inputs, outputs, score, labels, all_indexes = inputs.cpu().data.numpy(), outputs.cpu().data.numpy(), score.cpu().data.numpy(), labels.cpu().data.numpy(), all_indexes.cpu().data.numpy()
    
    merge_indexes, score = merge_repeat_records(score, all_indexes, mode="min")
    _, inputs = merge_repeat_records(inputs, all_indexes, mode="mean")
    _, outputs = merge_repeat_records(outputs, all_indexes, mode="mean")
    _, labels = merge_repeat_records(labels, all_indexes, mode="mean")
    
    return f_01.cpu().data.numpy(), f_1.cpu().data.numpy(), roc_auc, pr_auc, inputs, outputs, score, labels, merge_indexes

def get_anomaly_socre(args, model, dataloader, name="test"):

    model.eval()
    mean, covariance = load_mean_cov(args)

    outputs = []
    labels = []
    datas = []
    errors = []
    all_indexes = []

    for i, data in enumerate(dataloader):

        data_x, data_y, indexes = data
        data_x = data_x.to(args.device)
        data_y = data_y.to(args.device)
        indexes = indexes.to(args.device)

        _, output, error, _, _, _ = model(data_x, mode="test")

        output_idx = torch.arange(output.size(1)-1, -1, -1).to(args.device).long()
        reverse_output = output.index_select(1, output_idx)
        reverse_error = error.index_select(1, output_idx)

        datas.append(data_x.view(-1, args.ninp))
        labels.append(data_y.view(-1, 1))
        outputs.append(reverse_output.view(-1, args.ninp))
        errors.append(reverse_error.view(-1, args.ninp))
    
        all_indexes.append(indexes.view(-1, 1))
    
    outputs = torch.cat(outputs, dim=0)
    errors = torch.cat(errors, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze(dim=1)
    datas = torch.cat(datas, dim=0)
    all_indexes = torch.cat(all_indexes, dim=0)

    xm = (errors - mean)
    cov_eps = covariance + 1e-5 * torch.eye(covariance.size(0)).to(args.device)
    score = (xm).mm(cov_eps.inverse()) * xm
    score = score.sum(dim=1)
    
    return outputs, score, datas, labels, errors, all_indexes

def merge_repeat_records(scores, index, mode="min"):

    n = len(index)
    
    try:
        k = np.shape(scores)[1]
    except:
        k = 2
    
    res = np.zeros((n, k+1))
    for i in range(n):
        res[i,0] = index[i]
        res[i,1:] = scores[i]
        
    res = pd.DataFrame(res)
    
    # DIFF: enable calcuation on univariate data.
    cols = {}
    for i in range(0, k+1):
        cols.update({i:mode})   
    res = res.groupby(0).agg(cols)   

#     if mode=="min":
#         res = res.groupby(0).agg({0:'min', 1:'min', 2:'min'})
#     else:
#         res = res.groupby(0).agg({0:'mean', 1:'mean', 2:'mean'})

#     if mode=="min":
#         res = res.groupby(0).agg({0:'min', 1:'min'})
#     else:
#         res = res.groupby(0).agg({0:'mean', 1:'mean'})
        
    res = np.array(res)
    
    index = res[:,0]
    scores = res[:,1:]

    return index, scores   
    



