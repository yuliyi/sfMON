import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
from time import time
import logging
import random
import copy
import math
from collections import defaultdict
from scipy import stats
from math import sqrt
from radam import RAdam
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import DataStructs


def parse_args():
    parser = argparse.ArgumentParser(description="Run drug_side.")
    parser.add_argument('--data_dir', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--save_dir', nargs='?', default='./model/',
                        help='save_model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--log_dir', nargs='?', default='./log/',
                        help='Input data path.')
    parser.add_argument('--lr', type=float, default=0.001,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--n_epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=50,
                        help='Number of epoch for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.0000005,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--droprate', type=float, default=0.2,
                        metavar='FLOAT', help='dropout-rate')
    parser.add_argument('--batch_size', type=int, default=10240,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='GPU or CPU')

    args = parser.parse_args()
    return args

    # ----------------------------------------define log information--------------------------------------------------------


# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# --------------------------------------------model-------------------------------------------------------
class RegressionLoss(nn.Module):
    def __init__(self, device, lam=0.05, eps=0):
        super(RegressionLoss, self).__init__()
        self.lam = lam
        self.eps = torch.FloatTensor([eps]).to(device)
        # self.mse = nn.MSELoss()

    def forward(self, output, label):
        x0 = torch.where(label == 0)
        x1 = torch.where(label != 0)
        loss = torch.mean(torch.concatenate(((output[x1] - label[x1]) ** 2, self.lam * (output[x0] - label[x0]) ** 2)))
        return loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, device, alpha=.05, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        w = 1
        self.alpha = torch.tensor([alpha, w, w, w, w, w]).to(device)
        self.gamma = gamma

    def forward(self, preds, labels):
        labels = labels.type(torch.long)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        return loss.mean()


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.qw = nn.Linear(input_dim, input_dim)
        self.kw = nn.Linear(input_dim, input_dim)
        self.vw = nn.Linear(input_dim, input_dim)
        self.attn = nn.MultiheadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(input_dim // 2, input_dim))
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, Q):
        q = self.qw(Q)
        k = self.kw(X)
        v = self.vw(X)
        output = self.attn(q, k, v)[0]
        X = self.AN1(output)

        output = self.l1(X)
        X = self.AN2(output + Q)

        return X


class AE(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE, self).__init__()

        self.vector_size = vector_size

        self.encoder1 = nn.Sequential(nn.Linear(self.vector_size, self.vector_size // 2),
                                      nn.BatchNorm1d(self.vector_size // 2),
                                      nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Linear(self.vector_size // 2, self.vector_size // 3),
                                      nn.BatchNorm1d(self.vector_size // 3),
                                      nn.LeakyReLU())
        self.encoder3 = nn.Sequential(nn.Linear(self.vector_size // 3, self.vector_size // 6),
                                      nn.BatchNorm1d(self.vector_size // 6),
                                      nn.LeakyReLU())
        self.decoder3 = nn.Sequential(nn.Linear(self.vector_size // 6, self.vector_size // 3),
                                      nn.BatchNorm1d(self.vector_size // 3),
                                      nn.LeakyReLU())
        self.decoder2 = nn.Sequential(nn.Linear(self.vector_size // 3, self.vector_size // 2),
                                      nn.BatchNorm1d(self.vector_size // 2),
                                      nn.LeakyReLU())
        self.decoder1 = nn.Sequential(nn.Linear(self.vector_size // 2, self.vector_size),
                                      nn.BatchNorm1d(self.vector_size),
                                      nn.LeakyReLU())

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        h = self.encoder3(e2)
        d2 = self.decoder3(h)
        d1 = self.decoder2(d2)
        o = self.decoder1(d1)

        return h, o


class ConvNCF(nn.Module):
    def __init__(self, drug_features_matrix, side_features_matrix, drugs_dim, sides_dim, embed_dim, drug_feature_count, dropout, freq = 6):
        super(ConvNCF, self).__init__()

        self.drug_features_matrix = drug_features_matrix
        self.side_features_matrix = side_features_matrix
        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.freq = freq
        self.head = args.heads
        self.drug_feature_count = drug_feature_count
        self.single_dim = self.drugs_dim // self.drug_feature_count

        self.encoder = EncoderLayer(self.single_dim, self.head, self.dropout)
        self.layers = _get_clones(self.encoder, args.layers)
        self.ae = AE(self.drugs_dim)
        self.gru = nn.GRU(self.single_dim, self.single_dim // 2, num_layers=args.grus, bidirectional=True, dropout=self.dropout)
        self.Wd = nn.Sequential(nn.Linear(self.single_dim, self.embed_dim),
                                nn.BatchNorm1d(self.embed_dim),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.embed_dim, self.freq))

        self.Ws = nn.Sequential(nn.Linear(self.sides_dim, self.embed_dim),
                                nn.BatchNorm1d(self.embed_dim),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.embed_dim, self.freq))

        self.Md = nn.Sequential(nn.Linear(self.single_dim, self.embed_dim * 2),
                                nn.BatchNorm1d(self.embed_dim * 2),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.embed_dim * 2, self.embed_dim))

        self.Ms = nn.Sequential(nn.Linear(self.sides_dim, self.embed_dim * 2),
                                nn.BatchNorm1d(self.embed_dim * 2),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.embed_dim * 2, self.embed_dim))

        self.reg_layer = nn.Sequential(nn.Linear(self.freq - 1, (self.freq - 1) * 2),
                                       nn.BatchNorm1d((self.freq - 1) * 2),
                                       nn.LeakyReLU(),
                                       nn.Dropout(self.dropout),
                                       nn.Linear((self.freq - 1) * 2, self.freq),
                                       nn.BatchNorm1d(self.freq),
                                       nn.LeakyReLU(),
                                       nn.Dropout(self.dropout),
                                       nn.Linear(self.freq, 1))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, drug_indices, side_indices):
        x_drugs_hid, x_drug_raw = self.ae(self.drug_features_matrix)  # 3750----128
        drug_features = self.drug_features_matrix.reshape(len(self.drug_features_matrix), self.drug_feature_count, -1).permute(1, 0, 2)
        drug_features, _ = self.gru(drug_features)
        for layer in self.layers:
            drug_features = layer(drug_features, drug_features)
        drug_features = self.encoder(drug_features, x_drugs_hid.unsqueeze(0)).permute(1, 0, 2).flatten(1)
        x_drugs = drug_features + x_drugs_hid
        m_drugs = self.Md(x_drugs)
        m_sides = self.Ms(self.side_features_matrix)

        x_drugs = self.Wd(x_drugs)
        x_sides = self.Ws(self.side_features_matrix)  # 1988----128
        classification = torch.multiply(x_drugs[drug_indices], x_sides[side_indices])
        cls = torch.mm(m_drugs, m_sides.T)
        cls = cls[drug_indices, side_indices]

        freq = classification[:, 1:]
        freq = self.reg_layer(freq)

        return classification, cls, freq.squeeze(), x_drug_raw  # freq focal loss, binary, freq, drug_raw


def chunkIt(seq, num):
    data = []
    for i in range(0, len(seq), num):
        if i + num > len(seq):
            data.append(seq[i:])
        else:
            data.append(seq[i:i + num])

    return data


def early_stopping(model, epoch, best_epoch, valid_loss, best_loss, bad_counter):
    if valid_loss < best_loss:
        best_loss = valid_loss
        bad_counter = 0
        save_model(model, args.save_dir, epoch, best_epoch)
        best_epoch = epoch
    else:
        bad_counter += 1
    return bad_counter, best_loss, best_epoch


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    current_epoch = 0
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rd {}'.format(old_model_state_file))


def del_model():
    current_epoch = 0
    model_state_file = os.path.join(args.save_dir, 'model_epoch{}.pth'.format(current_epoch))
    os.remove(model_state_file)


def load_model(model, model_dir, best_epoch):
    best_epoch = 0
    model_path = os.path.join(model_dir, 'model_epoch{}.pth'.format(best_epoch))
    checkpoint = torch.load(model_path, map_location=get_device(args))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device


# -------------------------------------- metrics and evaluation define -------------------------------------------------
def Accuracy_micro(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        count += sum(np.logical_not(np.logical_xor(y_true[i], y_pred[i])))
    return count / y_true.size


def compute_mAP(y_true, y_pred):
    AP = average_precision_score(y_true, y_pred)
    return AP


def spearman(y_true, y_pred):
    sp = stats.spearmanr(y_true, y_pred)[0]
    return sp


def calc_metrics(pred_binary, y_binary, pred_freq, y_true, pred_mat=None, raw_mat=None, test_pos_mask=None):
    aupr = average_precision_score(y_binary, pred_binary)
    auc = metrics.roc_auc_score(y_binary, pred_binary)

    one_label_index = np.nonzero(y_true)
    sp = spearman(y_true[one_label_index], pred_freq[one_label_index])
    rmse = sqrt(mean_squared_error(y_true[one_label_index], pred_freq[one_label_index]))
    mae = mean_absolute_error(y_true[one_label_index], pred_freq[one_label_index])
    if pred_mat is not None:
        mAP, ndcg, p1, p15, r1, r15 = cal_map(pred_mat, raw_mat, test_pos_mask)
        return mAP, auc, aupr, ndcg, p1, p15, r1, r15, sp, rmse, mae
    return auc, aupr, sp, rmse, mae



def cal_map(total_preds, raw, mask):
    # others
    Tr_neg = {}
    Te = {}
    train_data = raw * mask
    Te_pairs = np.where(mask == 0)  # 测试集正样本
    Tr_neg_pairs = np.where(train_data == 0)  # 训练集正样本以外
    Te_pairs = np.array(Te_pairs).transpose()
    Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
    for te_pair in Te_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)

    for te_pair in Tr_neg_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Tr_neg:
            Tr_neg[drug_id] = [SE_id]
        else:
            Tr_neg[drug_id].append(SE_id)

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Tr_neg, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return map_value, ndcg, p1, p15, r1, r15


def evaluate_others(M, Tr_neg, Te, positions=[1, 5, 10, 15]):
    """
    :param M: 预测值
    :param Tr_neg: dict， 包含Te
    :param Te:  dict
    :param positions:
    :return:
    """
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0
    for u in Te:
        val = M[u, :]
        inx = np.array(Tr_neg[u])
        A = set(Te[u])
        B = set(inx) - A
        # compute precision and recall
        ii = np.argsort(val[inx])[::-1][:max(positions)]
        prec += precision(Te[u], inx[ii], positions)
        rec += recall(Te[u], inx[ii], positions)
        ndcg_user = nDCG(Te[u], inx[ii], 10)
        # compute map and AUC
        pos_inx = np.array(list(A))
        neg_inx = np.array(list(B))
        map_user, auc_user = map_auc(pos_inx, neg_inx, val)
        ndcg += ndcg_user
        map_value += map_user
        auc_value += auc_user
        # outf.write(" ".join([str(map_user), str(auc_user), str(ndcg_user)])+"\n")
    # outf.close()
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
        Te.keys()), rec / len(Te.keys())


def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):#nDCG(Te[u], inx[ii], 10)
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1/math.log(i+2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i]/math.log(i+2, 2) for i in range(num)])
    if idcg > 0:
        return dcg/idcg
    else:
        return idcg


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i,pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos<=neg:
                num+=1
            else:
                auc_num+=1
        map += (i+1)/(i+num+1)
    return map/len(pos_inx), auc_num/(len(pos_inx)*len(neg_inx))


def validate(model, drug_features_matrix, test_loader, device):
    model.eval()
    Classification_criterion = WeightedFocalLoss(device)
    ae_criterion = nn.MSELoss()
    regression_criterion = RegressionLoss(device)
    reg_loss = RegressionLoss(device)
    total_loss = 0
    with torch.no_grad():
        for test_drug, test_side, test_ratings in test_loader:
            pos_indices = test_ratings > 0
            out1, out2, out3, out4 = model(test_drug, test_side)
            loss1 = Classification_criterion(out1, test_ratings.to(device))
            loss2 = regression_criterion(out2, test_ratings.to(device))
            loss3 = reg_loss(out3[pos_indices], test_ratings[pos_indices].to(device))
            loss4 = ae_criterion(out4[test_drug], drug_features_matrix[test_drug].to(device))
            total_loss += loss1 + loss2 + loss3 + loss4
    return total_loss.item()


def evaluate(model, train_loader, valid_loader, test_loader, raw_mat, dn, sn, save=False, folder=None):
    model.eval()
    # binary classification (test set & other negative set)
    pred_binary = []
    y_binary = []
    # frequency regression
    pred_freq = []
    y_true = []
    # frequency result
    pred_result = np.zeros((0, 3))
    y_result = np.zeros((0, 3))
    # all sample predicted score calculated by lastest model
    pred_mat = np.zeros((dn, sn))
    test_pos_mask = np.ones((dn, sn))  # 测试集正样本位置设0，其他设1
    with torch.no_grad():
        for train_drug, train_side, train_ratings in train_loader:
            _, outputs_2, outputs_3, _ = model(train_drug, train_side)
            # pred_score = F.softmax(outputs_1, dim=-1).detach().cpu().numpy()
            pred_score = outputs_2.detach().cpu().numpy()
            freq_score = outputs_3.detach().cpu().numpy()
            pred_mat[train_drug, train_side] = pred_score
            neg_index = train_ratings == 0
            pred_binary = np.concatenate((pred_binary, pred_score[neg_index]), axis=0)
            pred_result = np.concatenate((pred_result, list(zip(train_drug[neg_index], train_side[neg_index], freq_score[neg_index]))), axis=0)
            y_result = np.concatenate((y_result, list(zip(train_drug[neg_index], train_side[neg_index], train_ratings.numpy()[neg_index]))), axis=0)

        for valid_drug, valid_side, valid_ratings in valid_loader:
            _, outputs_2, outputs_3, _ = model(valid_drug, valid_side)
            # pred_score = F.softmax(outputs_1, dim=-1).detach().cpu().numpy()
            pred_score = outputs_2.detach().cpu().numpy()
            freq_score = outputs_3.detach().cpu().numpy()
            pred_mat[valid_drug, valid_side] = pred_score
            neg_index = valid_ratings == 0
            pred_binary = np.concatenate((pred_binary, pred_score[neg_index]), axis=0)
            pred_result = np.concatenate((pred_result, list(zip(valid_drug[neg_index], valid_side[neg_index], freq_score[neg_index]))), axis=0)
            y_result = np.concatenate((y_result, list(zip(valid_drug[neg_index], valid_side[neg_index], valid_ratings.numpy()[neg_index]))), axis=0)
        y_binary = np.concatenate((y_binary, np.zeros(len(pred_binary))))

        for test_drug, test_side, test_ratings in test_loader:
            pos_indices = test_ratings > 0
            _, outputs_2, outputs_3, _ = model(test_drug, test_side)
            # pred_score = F.softmax(outputs_1, dim=-1).detach().cpu().numpy()
            pred_score = outputs_2.detach().cpu().numpy()
            freq_score = outputs_3.detach().cpu().numpy()
            pred_mat[test_drug, test_side] = pred_score
            test_pos_mask[test_drug[pos_indices], test_side[pos_indices]] = 0

            pred_binary = np.concatenate((pred_binary, pred_score), axis=0)
            y_binary = np.concatenate((y_binary, np.where(test_ratings.numpy() == 0, 0, 1)))
            pred_freq = np.concatenate((pred_freq, freq_score), axis=0)
            y_true = np.concatenate((y_true, test_ratings.numpy()), axis=0)

            pred_result = np.concatenate((pred_result, list(zip(test_drug, test_side, freq_score))), axis=0)
            y_result = np.concatenate((y_result, list(zip(test_drug, test_side, test_ratings.numpy()))), axis=0)

        if save:
            np.save(f"./result/pred_freq_result_{folder}.npy", pred_result)
            np.save(f"./result/y_freq_result_{folder}.npy", y_result)
            np.save(f"./result/pred_bin_result_{folder}.npy", pred_binary)
            np.save(f"./result/y_bin_result_{folder}.npy", y_binary)

    return calc_metrics(pred_binary, y_binary, pred_freq, y_true, pred_mat, raw_mat, test_pos_mask)


# ----------------------------------------------------------------------------------------------#

def Extract_positive_negative_samples(DAL):
    k = 0

    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)

    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]

    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    print("positive size:", (number_positive))
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    final_negtive_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]

    return final_positive_sample, final_negtive_sample


# -----------------------------------   train model  -------------------------------------------------------------------

def train(args):
    # seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and args.cuda else "cpu")

    # set log file
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)


    # initialize data
    with open('./data/drug_side.pkl', 'rb') as f:
        drug_side = pickle.load(f)

    # drug features
    drug_one = cosine_similarity(np.load('./data/gin_supervised_contextpred_750.npy'))
    drug_two = cosine_similarity(np.load('./data/gin_supervised_infomax_750.npy'))
    drug_three = cosine_similarity(np.load('./data/gin_supervised_edgepred_750.npy'))
    drug_four = cosine_similarity(np.load('./data/gin_supervised_masking_750.npy'))

    with open('./data/drug_fps.pkl', 'rb') as f:
        drug_fps = pickle.load(f)
    drug_five = np.identity(750)
    for i in range(750):
        for j in range(i + 1, 750):
            drug_five[i, j] = drug_five[j, i] = DataStructs.DiceSimilarity(drug_fps[i], drug_fps[j])

    with open('./data/drug_mols_vec_sim.pkl', 'rb') as f:
        drug_six = pickle.load(f)

    # side features
    gii = open('./data/ADR-994.pkl', 'rb')
    adr = pickle.load(gii)
    adr = np.array(adr)
    gii.close()

    drug_features, side_features = [], []
    drug_features.append(drug_one)
    drug_features.append(drug_two)
    drug_features.append(drug_three)
    drug_features.append(drug_four)
    drug_features.append(drug_five)
    drug_features.append(drug_six)
    drug_feature_count = 6
    side_features.append(adr)

    final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side)
    final_sample = np.vstack((final_positive_sample, final_negative_sample))  # 74774,3
    X = final_sample[:, 0::]
    data_x = []
    data_y = []

    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))

    drug_features_matrix = torch.from_numpy(np.hstack(drug_features)).type(torch.FloatTensor).to(device)
    side_features_matrix = torch.from_numpy(np.hstack(side_features)).type(torch.FloatTensor).to(device)

    # ====================   training    ====================
    # train model
    # use 10-fold cross validation
    all_p1_list = []
    all_p15_list = []
    all_r1_list = []
    all_r15_list = []
    all_auc_list = []
    all_aupr_list = []
    all_ndcg_list = []
    all_mAP_list = []
    all_Map_list = []
    all_sp_list = []
    all_rmse_list = []
    all_mae_list = []
    all_f1_list = []
    start_t = time()
    models = []
    for i in range(10):
        model = ConvNCF(drug_features_matrix, side_features_matrix, 750 * drug_feature_count, 768, args.embed_dim,
                        drug_feature_count, args.droprate)
        model.init_weights()
        model.to(device)
        models.append(model)
    kfold = StratifiedKFold(10, random_state=args.seed, shuffle=True)
    for idx, (train_index, test_index) in enumerate(kfold.split(data_x, data_y)):
        folder = idx + 1
        print(f"***********fold-{folder}***********")
        train_index, valid_index = train_test_split(train_index, test_size=0.05, random_state=args.seed)
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        c_x_train = data_x[train_index]
        c_y_train = data_y[train_index]
        c_x_valid = data_x[valid_index]
        c_y_valid = data_y[valid_index]
        c_x_test = data_x[test_index]
        c_y_test = data_y[test_index]
        trainset = torch.utils.data.TensorDataset(torch.IntTensor(c_x_train[:, 0]), torch.IntTensor(c_x_train[:, 1]),
                                                  torch.FloatTensor(c_y_train))
        validset = torch.utils.data.TensorDataset(torch.IntTensor(c_x_valid[:, 0]), torch.IntTensor(c_x_valid[:, 1]),
                                                  torch.FloatTensor(c_y_valid))
        testset = torch.utils.data.TensorDataset(torch.IntTensor(c_x_test[:, 0]), torch.IntTensor(c_x_test[:, 1]),
                                                 torch.FloatTensor(c_y_test))
        _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=2,
                                             pin_memory=True)
        _valid = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, num_workers=2,
                                             pin_memory=True)
        _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, num_workers=2,
                                            pin_memory=True)


        bad_counter = 0
        best_loss = 10000
        best_aupr = 0
        best_epoch = 0
        avg_loss = 0.0
        regression_criterion = RegressionLoss(device)
        reg_loss = RegressionLoss(device)
        classification_criterion = WeightedFocalLoss(device)
        ae_criterion = nn.MSELoss()
        optimizer = torch.optim.RAdam(models[idx].parameters(), lr=args.lr)

        time0 = time()
        for epoch in range(1, args.n_epoch + 1):
            models[idx].train()
            avg_loss = 0.0
            out_list = np.empty((0))
            y_binary = np.empty((0))
            freq_list = np.empty((0))
            y = np.empty((0))
            for i, data in enumerate(_train, 0):
                batch_drug, batch_side, batch_ratings = data
                pos_indices = batch_ratings > 0
                optimizer.zero_grad()
                out1, out2, out3, out4 = models[idx](batch_drug, batch_side)
                loss1 = classification_criterion(out1, batch_ratings.to(device))  # freq focal loss
                loss2 = regression_criterion(out2, batch_ratings.to(device))  # binary prediction
                loss3 = reg_loss(out3[pos_indices], batch_ratings[pos_indices].to(device))  # freq prediciton
                loss4 = ae_criterion(out4[batch_drug], drug_features_matrix[batch_drug].to(device))
                total_loss = loss1 + loss2 + loss3 + loss4
                total_loss.backward()
                optimizer.step()
                out_list = np.concatenate((out_list, out2.detach().cpu().numpy()), axis=0)
                y_binary = np.concatenate((y_binary, np.where(batch_ratings.numpy() == 0, 0, 1)))
                freq_list = np.concatenate((freq_list, out3.detach().cpu().numpy()), axis=0)
                y = np.concatenate((y, batch_ratings.numpy()), axis=0)
                avg_loss += total_loss.item()
            train_auc, train_aupr, train_sp, train_rmse, train_mae = calc_metrics(out_list, y_binary, freq_list, y)
            logging.info(
                'ADR Training: Folder:{}| Epoch:{}|loss:{:.4f}|AUC:{:.4f}|AUPR:{:.4f}|SP:{:.4f}|RMSE:{:.4f}|MAE:{:.4f}'.format(
                    folder, epoch, avg_loss, train_auc, train_aupr, train_sp, train_rmse, train_mae))
            valid_loss = validate(models[idx], drug_features_matrix, _valid, device)
            bad_counter, best_loss, best_epoch = early_stopping(models[idx], epoch, best_epoch, valid_loss, best_loss, bad_counter)
            logging.info('ADR Validation: Best_epoch {}|loss:{:.4f}'.format(best_epoch, valid_loss))
            if epoch % 10 == 9:
                mAP, auc, aupr, ndcg, p1, p15, r1, r15, sp, rmse, mae = evaluate(models[idx], _train, _valid, _test, drug_side, drug_side.shape[0], drug_side.shape[1])
                logging.info(
                    'ADR Evaluation: Best_epoch {}|mAP:{:.4f}|AUC:{:.4f}|AUPR:{:.4f}|ndcg:{:.4f}|P1:{:.4f}|P15:{:.4f}|R1:{:.4f}|R15:{:.4f}|SP:{:.4f}|RMSE:{:.4f}|MAE:{:.4f}'.format(
                        best_epoch, mAP, auc, aupr, ndcg, p1, p15, r1, r15, sp, rmse, mae))
            if best_epoch < 100 and bad_counter < 30:
                continue
            if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
                model = load_model(models[idx], args.save_dir, best_epoch)
                mAP, auc, aupr, ndcg, p1, p15, r1, r15, sp, rmse, mae = evaluate(model, _train, _valid, _test, drug_side, drug_side.shape[0], drug_side.shape[1], True, folder)
                logging.info(
                    'Final ADR Evaluation: Best_epoch {}|mAP:{:.4f}|AUC:{:.4f}|AUPR:{:.4f}|ndcg:{:.4f}|P1:{:.4f}|P15:{:.4f}|R1:{:.4f}|R15:{:.4f}|SP:{:.4f}|RMSE:{:.4f}|MAE:{:.4f}'.format(
                    best_epoch, mAP, auc, aupr, ndcg, p1, p15, r1, r15, sp, rmse, mae))
                all_p1_list.append(p1)
                all_p15_list.append(p15)
                all_r1_list.append(r1)
                all_r15_list.append(r15)
                all_aupr_list.append(aupr)
                all_mAP_list.append(mAP)
                all_auc_list.append(auc)
                all_ndcg_list.append(ndcg)
                all_sp_list.append(sp)
                all_mae_list.append(mae)
                all_rmse_list.append(rmse)
                break
    mean_mAP = np.mean(all_mAP_list)
    mean_p1 = np.mean(all_p1_list)
    mean_p15 = np.mean(all_p15_list)
    mean_r1 = np.mean(all_r1_list)
    mean_r15 = np.mean(all_r15_list)
    mean_aupr = np.mean(all_aupr_list)
    mean_auc = np.mean(all_auc_list)
    mean_ndcg = np.mean(all_ndcg_list)
    mean_mae = np.mean(all_mae_list)
    mean_sp = np.mean(all_sp_list)
    mean_rmse = np.mean(all_rmse_list)
    logging.info(
        '10-fold cross validation DDI Mean Evaluation: Total Time {:.1f}s|mAP:{:.4f}|AUC:{:.4f}|AUPR:{:.4f}|ndcg:{:.4f}|P1:{:.4f}|P15:{:.4f}|R1:{:.4f}|R15:{:.4f}|SP:{:.4f}|RMSE:{:.4f}|MAE:{:.4f}'.format(
            time() - start_t, mean_mAP, mean_auc, mean_aupr, mean_ndcg, mean_p1, mean_p15, mean_r1, mean_r15, mean_sp, mean_rmse, mean_mae))


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    args.heads = 6
    args.loss_weight = 2
    args.layers = 1
    args.grus = 1
    train(args)
