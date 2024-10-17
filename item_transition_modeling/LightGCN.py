#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def bpr_loss(pos_score, neg_score):
    loss = - F.logsigmoid(pos_score - neg_score)
    loss = torch.mean(loss)
    return loss


def laplace_transform(graph):
    # laplace transform
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    # to tensor
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class LightGCN(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.emb_size = conf["emb_size"]
        self.num_nodes = conf["num_nodes"]

        self.node_features = self.init_embeddings()

        self.raw_graph = raw_graph
        self.norm_graph_ori = self.get_norm_graph_ori()
        self.norm_graph = self.get_norm_graph()

        self.num_layers = self.conf["num_layers"]
        # Dropouts
        self.mess_dropout = nn.Dropout(self.conf["aug_rate"], True)


    def init_embeddings(self):
        node_features = nn.Parameter(torch.FloatTensor(self.num_nodes, self.emb_size))
        nn.init.xavier_normal_(node_features)
        return node_features


    def get_norm_graph_ori(self):
        norm_graph_ori = laplace_transform(self.raw_graph).to(self.device)
        return norm_graph_ori


    def get_norm_graph(self):
        raw_graph = self.raw_graph
        device = self.device
        aug_rate = self.conf['aug_rate']
        if aug_rate != 0:
            if self.conf["aug_type"] == "ED":
                graph = raw_graph.tocoo()
                values = np_edge_dropout(graph.data, aug_rate)
                raw_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        norm_graph = laplace_transform(raw_graph).to(device)
        return norm_graph


    def one_propagate(self, graph, node_feature, mess_dropout, test):
        features = node_feature
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                features = mess_dropout(features)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.mean(all_features, dim=1).squeeze(1)

        return all_features


    def propagate(self, test=False):
        if test:
            node_representations = self.one_propagate(self.norm_graph_ori, self.node_features, self.mess_dropout, test)
        else:
            node_representations = self.one_propagate(self.norm_graph, self.node_features, self.mess_dropout, test)

        return node_representations


    def predict(self, A_feature, B_feature):
        pred = torch.sum(A_feature * B_feature, 1)
        return pred


    def forward(self, batch, ED_drop=False):
        # apply the edge dropout to generate a ub_graph.
        if ED_drop:
            self.norm_graph = self.get_norm_graph()

        a_id, pos_id, neg_id = batch
        node_representations = self.propagate()
        node_rep = node_representations[a_id]
        pos_rep = node_representations[pos_id]
        neg_rep = node_representations[neg_id]

        pos_pred = self.predict(node_rep, pos_rep)
        neg_pred = self.predict(node_rep, neg_rep)
        loss = bpr_loss(pos_pred, neg_pred)
        return loss


    def evaluate(self, propagate_result, target_ids):
        node_representations = propagate_result
        target_embeddings = node_representations[target_ids]
        scores = torch.mm(target_embeddings, node_representations.t())

        return scores
