#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import torch
import tqdm
import matplotlib.pyplot as plt
from item_transition_modeling.utils import Datasets
from item_transition_modeling.LightGCN import LightGCN
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import random



def load_GNN_from_checkpoints(conf_path, ckpt_path, device=None):
    with open(conf_path, "r") as f:
        temp_conf = json.load(f)
    # assert temp_conf['split_data'] == False

    dataset = Datasets(temp_conf)
    if device is None:
        device = torch.device("cpu")

    temp_conf["device"] = device
    model_name = temp_conf['model']
    # model
    if model_name == "LightGCN":
        model = LightGCN(temp_conf, dataset.adj_graph_train)
    else:
        raise Exception(f"Unimplemented model type {model_name}")

    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model = model.to(device)
    model.eval()
    return model


# Given dataset path, the function loads and returns the mappings between source_ids and item_ids
def load_item_id_mappings(dataset_path):
    mapping_path = dataset_path + 'item_source_id_mapping.json'
    with open(mapping_path, "r") as f:
        mappings = json.load(f)

    source_to_id_mappings = {}
    id_to_source_mappings = {}

    for item in mappings.items():
        item_id, source_id = item
        item_id = int(item_id)

        source_to_id_mappings[source_id] = item_id
        id_to_source_mappings[item_id] = source_id

    return source_to_id_mappings, id_to_source_mappings


class GNN_Graph_Encoder():
    def __init__(self, dataset_path, model):
        super().__init__()
        self.dataset_path = dataset_path
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.node_reps = self.generate_item_representations()

        self.source_to_id_mappings, self.id_to_source_mappings = load_item_id_mappings(dataset_path)
        # self.self_quality_evaluation()


    def generate_item_representations(self):
        node_reps = self.model.propagate(test=True)
        return node_reps
    

    # drop the edges with lowest scores evaluted by the teacher model
    def drop_graph(self, drop_rate, input_edges, input_graph=None):
        if not isinstance(input_edges, torch.Tensor):
            input_edges = torch.tensor(input_edges, dtype=torch.long)

        edge_reps = self.node_reps[input_edges]
        edge_scores = torch.sum(edge_reps[:, 0] * edge_reps[:, 1], dim=1).detach().cpu().numpy()
        sorted_indices = np.argsort(edge_scores)
        # score the edges in ascending order according to similarity scores.
        sorted_input_edges = input_edges[sorted_indices]
        
        separation_index = int(edge_scores.shape[0] * drop_rate)
        low_score_edges = sorted_input_edges[:separation_index]
        high_score_edges = sorted_input_edges[separation_index:]

        if input_graph is not None:
            remaining_graph = input_graph.copy()
            for drop_edge in low_score_edges:
                i, j = drop_edge
                remaining_graph[i, j] -= 1
            
            remaining_graph.eliminate_zeros()
            return high_score_edges, low_score_edges, remaining_graph
        
        else:
            return high_score_edges, low_score_edges
        
    
    # For calculating the pairwise similarity, there is a more efficient way of implementing this.
    def popularity_preserved_drop_graph(self, drop_rate, input_edges, input_graph):
        dropped_graph = input_graph.copy()
        num_nodes = dropped_graph.shape[0]
        for node_i in tqdm.tqdm(range(num_nodes)):
            node_i_similarities = torch.matmul(self.node_reps[node_i], self.node_reps.T)
            ground_truth_nodes = dropped_graph[node_i].indices
            ground_truth_scores = node_i_similarities[ground_truth_nodes]

            # determine number of edges to drop while preserving the popularity distribution.
            expected_drop = ground_truth_scores.shape[0] * drop_rate
            additional_drop_prob = expected_drop - int(expected_drop)
            edges_to_drop = int(expected_drop) + (random.random() < additional_drop_prob)

            _, lowest_indices = torch.topk(ground_truth_scores, edges_to_drop, largest=False)
            lowest_indices = lowest_indices.detach().cpu().numpy()
            nodes_to_drop = ground_truth_nodes[lowest_indices]

            for node_j in nodes_to_drop:
                dropped_graph[node_i, node_j] = 0
        
        dropped_graph.eliminate_zeros()
        print(f"Graph dropped complete. The original graph contains {input_graph.data.sum()} edges. {dropped_graph.data.sum()} edges remain after dropping with rate {drop_rate}.")

        # generate the dropped edge set accroding to the csr_matrix.
        rows, cols = dropped_graph.nonzero()
        repeat_counts = dropped_graph.data.astype(int)
        edge_rows = np.repeat(rows, repeat_counts)
        edge_cols = np.repeat(cols, repeat_counts)
        dropped_edges = np.vstack((edge_rows, edge_cols)).T

        return dropped_edges, dropped_graph
    

    def get_high_score_edges(self, add_rate, input_edges, input_graph):
        ori_graph = input_graph.copy()
        num_nodes = ori_graph.shape[0]

        generated_high_score_edges = []
        for node_i in tqdm.tqdm(range(num_nodes)):
            node_i_similarities = torch.matmul(self.node_reps[node_i], self.node_reps.T)
            ground_truth_nodes = ori_graph[node_i].indices

            # determine number of edges to drop while preserving the popularity distribution.
            expected_num_add = ground_truth_nodes.shape[0] * add_rate
            additional_add_prob = expected_num_add - int(expected_num_add)
            edges_to_add = int(expected_num_add) + (random.random() < additional_add_prob)

            _, highest_indices = torch.topk(node_i_similarities, edges_to_add, largest=True)
            for node_j in highest_indices:
                generated_high_score_edges.append([node_i, int(node_j)])
        
        generated_high_score_edges = np.array(generated_high_score_edges)

        print(f"High score edges generated successfully. The original graph contains {input_graph.data.sum()} edges. {generated_high_score_edges.shape[0]} edges generated under rate {add_rate}.")

        # generate the dropped edge set accroding to the csr_matrix.
        indice = np.array(generated_high_score_edges, dtype=np.int32)
        values = np.ones(len(generated_high_score_edges), dtype=np.float32)
        generated_high_score_graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(num_nodes, num_nodes)).tocsr()

        return generated_high_score_edges, generated_high_score_graph
    