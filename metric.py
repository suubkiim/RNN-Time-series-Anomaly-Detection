import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, cohen_kappa_score



def get_precison_recall(score, label, sz, beta=1.0):
    
    maximum = score.max().item()
    th = torch.linspace(0, maximum, sz)

    precison = []
    recall = []

    for i in range(len(th)):
        anomaly = (score > th[i]).float()
        idx = anomaly * 2 + label
        tn = (idx==0).sum().item()
        fn = (idx==1).sum().item()
        fp = (idx==2).sum().item()
        tp = (idx==3).sum().item()

        p = tp/(tp+fp+1e-7)
        r = tp/(tp+fn+1e-7)

        if p!=0 and r!=0:
            precison.append(p)
            recall.append(r)
    
    precison = torch.Tensor(precison)
    recall = torch.Tensor(recall)

    f1 = (1+beta**2)*torch.max((precison*recall).div(beta**2*precison+recall+1e-7))

    return precison, recall, f1


def CalculateROCAUCMetrics(_score, _abnormal_label):

    _score = _score.cpu().data.numpy()
    _abnormal_label = _abnormal_label.cpu().data.numpy()

    fpr, tpr, _ = roc_curve(_abnormal_label, _score)
    roc_auc = auc(np.nan_to_num(fpr), np.nan_to_num(tpr))

    return fpr, tpr, roc_auc


def CalculatePrecisionRecallCurve(_score, _abnormal_label):

    _score = _score.cpu().data.numpy()
    _abnormal_label = _abnormal_label.cpu().data.numpy()
    
    precision_curve, recall_curve, _ = precision_recall_curve(_abnormal_label, _score)
    average_precision = average_precision_score(_abnormal_label, _score)

    return precision_curve, recall_curve, average_precision

def adjust_learning_rate(args, optimizer, epoch, gammas, schedule):

    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr