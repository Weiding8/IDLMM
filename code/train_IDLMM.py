import sys
import os
import numpy as np
import scipy
import torch
import torch.utils.data as du
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from IDLMM import *
import argparse
import numpy as np
import time

def create_term_mask(term_direct_gene_map, gene_dim):
	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), gene_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

		term_mask_map[term] = mask_gpu

	return term_mask_map


def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder,
				train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug,num_hiddens_jjh, num_hiddens_final,
				cell_features, drug_features, jjh_dim, nhead, num_layers):
	epoch_start_time = time.time()
	best_model = 0
	max_corr = 0

	# dcell neural network
	model = IDLMM_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype,
						num_hiddens_drug,num_hiddens_jjh, num_hiddens_final, jjh_dim, nhead, num_layers)

	train_feature, train_label, test_feature, test_label = train_data

	train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
	test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

	model.cuda(CUDA_ID)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

	optimizer.zero_grad()

	for name, param in model.named_parameters():
		term_name = name.split('_')[0]

		if '_direct_gene_layer.weight' in name:
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
		else:
			param.data = param.data * 0.1

	train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=batch_size, shuffle=False)
	test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=False)


	for epoch in range(train_epochs):

		# Train
		model.train()
		train_predict = torch.zeros(0, 0).cuda(CUDA_ID)

		for i, (inputdata, labels) in enumerate(train_loader):
			# Convert torch tensor to Variable
			features = build_input_vector(inputdata, cell_features, drug_features)

			cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
			cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer

			# Here term_NN_out_map is a dictionary
			aux_out_map, _, att = model(cuda_features)
			# print(att)

			if train_predict.size()[0] == 0:
				train_predict = aux_out_map['final'].data
			else:
				train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

			total_loss = 0
			for name, output in aux_out_map.items():
				loss = nn.MSELoss()
				if name == 'final':
					total_loss += loss(output, cuda_labels)
				else:  
					total_loss += 0.01 * loss(output, cuda_labels)

			total_loss.backward()

			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]
				# print name, param.grad.data.size(), term_mask_map[term_name].size()
				param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

			optimizer.step()

		train_corr = pearson_corr(train_predict, train_label_gpu)
		
		model.eval()

		# Test: random variables in training mode become static


		test_predict = torch.zeros(0, 0).cuda(CUDA_ID)

		for i, (inputdata, labels) in enumerate(test_loader):
			# Convert torch tensor to Variable
			features = build_input_vector(inputdata, cell_features, drug_features)
			cuda_features = Variable(features.cuda(CUDA_ID))

			aux_out_map, _, att = model(cuda_features)

			if test_predict.size()[0] == 0:
				test_predict = aux_out_map['final'].data
			else:
				test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		test_corr = pearson_corr(test_predict, test_label_gpu)
		

		epoch_end_time = time.time()

		print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s" % (
		epoch, CUDA_ID, train_corr, test_corr, total_loss, epoch_end_time - epoch_start_time))

		epoch_start_time = epoch_end_time

		if test_corr >= max_corr:
			max_corr = test_corr
			best_model = epoch
			torch.save(model, model_save_folder + '/IDLMM_best.pt')

	torch.save(model, model_save_folder + '/IDLMM_500.pt')
    

	print("Best performed model (epoch)\t%d" % best_model)


parser = argparse.ArgumentParser(description='Train IDLMM')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str, default='../data/ont.txt')
parser.add_argument('-train', help='Training dataset', type=str, default='../data/a_train_data.txt')
parser.add_argument('-test', help='Validation dataset', type=str, default='../data/a_valid_data.txt')
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=500)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default='../data/gene2ind.txt')
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default='../data/drug_id.txt')
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default='../data/cell_id.txt')

parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts',
					type=int, default=6)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str,
					default='100,50,6')
parser.add_argument('-jjh_hiddens', help='The number of neurons in the jjh part', type=int, default=6)
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=6)

parser.add_argument('-nhead', help='The number of attention heads', type=int, default=4)
parser.add_argument('-num_layers', help='The number of TransformerEncoder layers', type=int, default=1)
# parser.add_argument('-output_dim', help='The number of Transformer output dim', type=int, default=6)

parser.add_argument('-genotype', help='Mutation information for cell lines', type=str, default='../data/mut.txt')
parser.add_argument('-cn', help='cn information for cell lines', type=str, default='../data/cn.txt')
parser.add_argument('-exp', help='exp information for cell lines', type=str, default='../data/exp.txt')
parser.add_argument('-jjh', help='jjh information for cell lines', type=str, default='../data/cpg.txt')
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str, default='../data/drugfingerprint.txt')

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.test, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.genotype, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

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

jjh_dim = len(cell_jjh[0, :])

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))
num_hiddens_jjh = opt.jjh_hiddens
num_hiddens_final = opt.final_hiddens
nhead = opt.nhead
num_layers = opt.num_layers
# output_dim = opt.output_dim
#####################################

CUDA_ID = opt.cuda

train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, opt.modeldir, opt.epoch,
			opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug,num_hiddens_jjh, num_hiddens_final, cell_features,
            drug_features, jjh_dim, nhead, num_layers)