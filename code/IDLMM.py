import argparse
import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(MyTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 调用自注意力层并返回注意力权重
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 继续前向传播
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # 返回输出和注意力权重


class IDLMM_nn(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, num_hiddens_genotype,
                 num_hiddens_drug, num_hiddens_jjh, num_hiddens_final, jjh_dim, nhead, num_layers):  # TODO

        super(IDLMM_nn, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug
        self.num_hiddens_jjh = num_hiddens_jjh

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes
        self.gene_dim = ngene
        self.drug_dim = ndrug
        # TODO
        self.jjh_dim = jjh_dim

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        self.construct_NN_drug()

        # TODO
        self.add_module('jjh_encoder_layer', MyTransformerEncoderLayer(d_model=32, nhead=nhead))
        # self.add_module('jjh_transformer_encoder',
        #                 nn.TransformerEncoder(self._modules['jjh_encoder_layer'], num_layers=num_layers))

        self.add_module('jjh_linear_layer', nn.Linear(32, num_hiddens_jjh))

        self.add_module('jjh_batchnorm_layer', nn.BatchNorm1d(num_hiddens_jjh))
        self.add_module('jjh_aux_linear_layer1', nn.Linear(num_hiddens_jjh, 1))
        self.add_module('jjh_aux_linear_layer2', nn.Linear(1, 1))


        # add modules for final layer
        final_input_size_gene = num_hiddens_genotype + num_hiddens_drug[-1]
        self.add_module('final1_linear_layer', nn.Linear(final_input_size_gene, num_hiddens_final))
        self.add_module('final1_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final1_aux_linear_layer', nn.Linear(num_hiddens_final, 1))
        self.add_module('final1_linear_layer_output', nn.Linear(1, 1))

        final_input_size_jjh = num_hiddens_jjh + num_hiddens_drug[-1]
        self.add_module('final2_linear_layer', nn.Linear(final_input_size_jjh, num_hiddens_final))
        self.add_module('final2_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final2_aux_linear_layer', nn.Linear(num_hiddens_final, 1))
        self.add_module('final2_linear_layer_output', nn.Linear(1, 1))

        # TODO
        # self.add_module('finally_output', nn.Linear(2, 1))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output

    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
            self.add_module(term + '_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))

    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        for i in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(i + 1), nn.Linear(input_size, self.num_hiddens_drug[i]))
            self.add_module('drug_batchnorm_layer_' + str(i + 1), nn.BatchNorm1d(self.num_hiddens_drug[i]))
            self.add_module('drug_aux_linear_layer1_' + str(i + 1), nn.Linear(self.num_hiddens_drug[i], 1))
            self.add_module('drug_aux_linear_layer2_' + str(i + 1), nn.Linear(1, 1))

            input_size = self.num_hiddens_drug[i]

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]
            # leaves = [n for n,d in dG.out_degree().items() if d==0]
            # leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term + '_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term + '_aux_linear_layer1', nn.Linear(term_hidden, 1))
                self.add_module(term + '_aux_linear_layer2', nn.Linear(1, 1))

            dG.remove_nodes_from(leaves)

    # definition of forward function
    def forward(self, x):
        gene_input = x.narrow(1, 0, self.gene_dim)
        # TODO
        jjh_input = x.narrow(1, self.gene_dim, self.jjh_dim)
        drug_input = x.narrow(1, self.gene_dim + self.jjh_dim, self.drug_dim)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        term_NN_out_map = {}
        aux_out_map = {}

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)

                term_NN_out = self._modules[term + '_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term + '_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term + '_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_aux_linear_layer2'](aux_layer1_out)

        # TODO

        # define forward function for jjh  #################################################

        jjh_out = jjh_input

        batch_size = jjh_input.size(0)
        jjh_out = jjh_input.view(batch_size, 44, -1)  # (batch_size, 16, 101)
        jjh_out = jjh_out.permute(1, 0, 2)  # (16, batch_size, 101)
        # Transformer 编码器
        # jjh_out,att = self._modules['jjh_transformer_encoder'](jjh_out)  # (16, batch_size, 101)
        jjh_out, att = self._modules['jjh_encoder_layer'](jjh_out)

        # 全局均值池化
        jjh_out = jjh_out.mean(dim=0)  # (batch_size, 16)

        jjh_out = self._modules['jjh_batchnorm_layer'](
            torch.tanh(self._modules['jjh_linear_layer'](jjh_out)))

        term_NN_out_map['jjh'] = jjh_out
        aux_layer1_out = torch.tanh(self._modules['jjh_aux_linear_layer1'](jjh_out))
        aux_out_map['jjh'] = self._modules['jjh_aux_linear_layer2'](aux_layer1_out)

        # define forward function for drug dcell #################################################
        drug_out = drug_input
        for i in range(1, len(self.num_hiddens_drug) + 1, 1):
            drug_out = self._modules['drug_batchnorm_layer_' + str(i)](
                torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_' + str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules['drug_aux_linear_layer1_' + str(i)](drug_out))
            aux_out_map['drug_' + str(i)] = self._modules['drug_aux_linear_layer2_' + str(i)](aux_layer1_out)

        # connect two neural networks at the top #################################################
        final_input_gene = torch.cat((term_NN_out_map[self.root], drug_out), 1)
        out1 = self._modules['final1_batchnorm_layer'](
            torch.tanh(self._modules['final1_linear_layer'](final_input_gene)))
        term_NN_out_map['final1'] = out1
        aux_layer_out = torch.tanh(self._modules['final1_aux_linear_layer'](out1))
        aux_out_map['final1'] = self._modules['final1_linear_layer_output'](aux_layer_out)

        final_input_jjh = torch.cat((jjh_out, drug_out), 1)
        out2 = self._modules['final2_batchnorm_layer'](
            torch.tanh(self._modules['final2_linear_layer'](final_input_jjh)))
        term_NN_out_map['final2'] = out2
        aux_layer_out = torch.tanh(self._modules['final2_aux_linear_layer'](out2))
        aux_out_map['final2'] = self._modules['final2_linear_layer_output'](aux_layer_out)

        # output = torch.cat((aux_out_map['final1'], aux_out_map['final2']), 1)
        # output = self._modules['finally_output'](output)
        output = (aux_out_map['final1'] + aux_out_map['final2']) / 2
        aux_out_map['final'] = output
        # aux_out_map['final'] = output

        return aux_out_map, term_NN_out_map, att

