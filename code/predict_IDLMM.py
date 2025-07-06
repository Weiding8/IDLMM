import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from IDLMM import *
import argparse
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def predict_IDLMM(predict_data, gene_dim, drug_dim, model_file, hidden_folder, batch_size, result_file, cell_features,
                  drug_features):
    # feature_dim = gene_dim + drug_dim

    model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

    predict_feature, predict_label = predict_data

    predict_label_gpu = predict_label.cuda(CUDA_ID)

    model.cuda(CUDA_ID)
    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

    # Test
    test_predict = torch.zeros(0, 0).cuda(CUDA_ID)
    att_weight = torch.zeros(0, 0).cuda(CUDA_ID)

    term_hidden_map = {}

    batch_num = 0
    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        features = build_input_vector(inputdata, cell_features, drug_features)

        cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)

        # make prediction for test data
        aux_out_map, term_hidden_map, att = model(cuda_features)
        # print(att)

        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['final'].data
        else:
            test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

        if att_weight.size()[0] == 0:
            att_weight = att
        else:
            att_weight = torch.cat([att_weight, att], dim=0)

        # for term, hidden_map in term_hidden_map.items():
        # 	hidden_file = hidden_folder+'/'+term+'.hidden'
        # 	with open(hidden_file, 'ab') as f:
        # 		np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

        batch_num += 1

    # print(att_weight.shape)
    # print(att_weight)

    test_corr = pearson_corr(test_predict, predict_label_gpu)
    print("Test pearson corr\t%.6f" % (test_corr))
    # print(term_hidden_map[model.root])
    #
    # np.savetxt(result_file + '/IDLMM.predict', test_predict.cpu().numpy(), '%.4e')







   


parser = argparse.ArgumentParser(description='Train IDLMM')
parser.add_argument('-predict', help='Dataset to be predicted', type=str,
                    default='../data/a_test_data.txt')
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default='../data/gene2ind.txt')
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default='../data/drug_id.txt')
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default='../data/cell_id.txt')

parser.add_argument('-load', help='Model file', type=str, default='MODEL/mymodel_JZ/drugcell_transEncoder1408_4432_att_500_sorted_JZ.pt')
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='../code/Hidden/')
parser.add_argument('-result', help='Result file name', type=str, default='Result/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)

parser.add_argument('-genotype', help='Mutation information for cell lines', type=str, default='../data/mut.txt')
parser.add_argument('-cn', help='cn information for cell lines', type=str, default='../data/cn.txt')
parser.add_argument('-exp', help='exp information for cell lines', type=str, default='../data/exp.txt')
parser.add_argument('-jjh', help='jjh information for cell lines', type=str, default='../data/cpg.txt')#cpg_1408.txt

parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str,
                    default='../data/drugfingerprint.txt')

opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_mutation = np.genfromtxt(opt.genotype, delimiter=',')
cell_exp = np.genfromtxt(opt.exp, delimiter=',')
cell_cn = np.genfromtxt(opt.cn, delimiter=',')

cell_jjh = np.genfromtxt(opt.jjh, delimiter=',')

drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

min_max_scaler = MinMaxScaler()
scaler = StandardScaler()
exp_result = min_max_scaler.fit_transform(cell_exp)
cell_exp = np.around(exp_result, 5)

cn_result = min_max_scaler.fit_transform(cell_cn)
cell_cn = np.around(cn_result, 5)


jjh_result = min_max_scaler.fit_transform(cell_jjh)
cell_jjh = np.around(jjh_result, 5)

gene_features = cell_mutation + cell_exp + cell_cn
cell_features = np.concatenate((gene_features, cell_jjh), axis=1)


num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0, :])

CUDA_ID = opt.cuda

print("Total number of genes = %d" % num_genes)

predict_IDLMM(predict_data, num_genes, drug_dim, opt.load, opt.hidden, opt.batchsize, opt.result, cell_features,
              drug_features)
