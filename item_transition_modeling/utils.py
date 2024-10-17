import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import json
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, data_name):
    print('>' * 10 + data_name + '>' * 10)
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class TrainData(Dataset):
    def __init__(self, adj_pairs, adj_graph_train):
        self.adj_pairs = adj_pairs
        self.adj_graph = adj_graph_train
        self.num_nodes = adj_graph_train.shape[0]

    def __len__(self):
        return len(self.adj_pairs)

    def __getitem__(self, idx):
        source, pos_end = self.adj_pairs[idx]
        neg_end = np.random.randint(self.num_nodes)
        while self.adj_graph[source, neg_end] == 1:
            neg_end = np.random.randint(self.num_nodes)
        return int(source), int(pos_end), int(neg_end)
    

# Return neighboring sub-graph of the indicated node.
# $graph - (scipy.sparse.csr_matrix): full graph to be sampled on. 
# $node_id - (numpy.ndarray of shape [N]): Node indices to be sampled.
# $num_samples - (list of integers): Number of neighbors sampled from each level of the neighbors.
def sample_nodes_sub_graph(graph, node_ids, num_samples, neighbor_level):
    # create id lookup matrix in sparse csr matrix
    num_batch_nodes = node_ids.shape[0]
    row = np.arange(num_batch_nodes)
    col = node_ids
    values = np.ones(num_batch_nodes)
    current_node_list = sp.csr_matrix((values, (row, col)), shape=(num_batch_nodes, graph.shape[1]))

    # node ids of all neighbors at each level
    order_neighbors = []
    for i in range(neighbor_level):
        neighbors = current_node_list @ graph
        # sampling should be performed after each sub-graph to decrease computation.
        neighbors = neighbor_sampling(neighbors, num_samples[i])
        order_neighbors.append(neighbors)
        current_node_list = neighbors

    # for i in range(neighbor_level):
    #     order_neighbors[i] = neighbor_sampling(order_neighbors[i], num_samples[i])
    return order_neighbors


# Masks the input neighboring adjacent matrix so that each node at maximum of {max_num_neighbors} neighbors.
# Note: this function performs an inplace operation.
# $neighbors - (scipy.sparse.csr_matrix) of shape [batch_size, num_nodes]
def neighbor_sampling(neighbors, max_num_neighbors=10):
    num_rows = neighbors.shape[0]
    for row_i in range(num_rows):
        start_index = neighbors.indptr[row_i]
        end_index = neighbors.indptr[row_i + 1]

        # for rows with more than $max_num_neighbors non-zero elements, randomly mask elements with $max_num_neighbors remaining elements
        num_nnz = end_index - start_index
        if num_nnz > max_num_neighbors:
            mask = np.concatenate((np.ones(max_num_neighbors, dtype=np.int32), np.zeros(num_nnz - max_num_neighbors, dtype=np.int32)), axis=0)
            np.random.shuffle(mask)
            neighbors.data[start_index: end_index] *= mask

    neighbors.eliminate_zeros()
    return neighbors
    

class TrainDataGraphBatching(Dataset):
    def __init__(self, adj_pairs, adj_graph_train, num_samples, neighbor_level):
        self.adj_pairs = adj_pairs
        self.adj_graph = adj_graph_train
        self.num_nodes = adj_graph_train.shape[0]
        self.num_samples = num_samples
        self.neighbor_level = neighbor_level

    def __len__(self):
        return len(self.adj_pairs)

    def __getitem__(self, idx):
        source, pos_end = self.adj_pairs[idx]
        neg_end = np.random.randint(self.num_nodes)
        while self.adj_graph[source, neg_end] == 1:
            neg_end = np.random.randint(self.num_nodes)
        return int(source), int(pos_end), int(neg_end)
    
    # collate function that returns sampled sub-graphs for nodes inside one batch input.
    def collate_sampled_sub_graphs(self, batch):
        source_ids, pos_end_ids, neg_end_ids = zip(*batch)
        source_ids = np.array(source_ids)
        pos_end_ids = np.array(pos_end_ids)
        neg_end_ids = np.array(neg_end_ids)
        
        source_order_sub_graphs = sample_nodes_sub_graph(self.adj_graph, source_ids, self.num_samples, self.neighbor_level)
        pos_order_end_sub_graphs = sample_nodes_sub_graph(self.adj_graph, pos_end_ids, self.num_samples, self.neighbor_level)
        neg_order_end_sub_graphs = sample_nodes_sub_graph(self.adj_graph, neg_end_ids, self.num_samples, self.neighbor_level)

        return (source_ids, pos_end_ids, neg_end_ids), (source_order_sub_graphs, pos_order_end_sub_graphs, neg_order_end_sub_graphs)
    
    # Return the sampled neighbors of the full training grpah.
    # Used for static performance evaluation. 
    def yeild_sampled_full_graph(self):
        full_node_ids = np.arange(self.num_nodes)
        sampled_full_graphs = sample_nodes_sub_graph(self.adj_graph, full_node_ids, self.num_samples, self.neighbor_level)
        return full_node_ids, sampled_full_graphs


class TestData(Dataset):
    def __init__(self, adj_pairs_test, adj_graph_test, adj_graph_train):
        self.adj_pairs_test = adj_pairs_test
        self.adj_graph_test = adj_graph_test
        self.adj_graph_train = adj_graph_train
        self.num_nodes = self.adj_graph_train.shape[0]

    def __len__(self):
        return len(self.adj_pairs_test)

    def __getitem__(self, idx):
        ground_truth_pair = self.adj_pairs_test[idx]
        source_id = ground_truth_pair[0]

        target_id = np.zeros(self.num_nodes)
        target_id[int(ground_truth_pair[1])] = 1

        train_mask = torch.from_numpy(self.adj_graph_train[source_id].toarray()).squeeze()

        return int(source_id), target_id, train_mask
    


class Datasets():
    def __init__(self, conf):
        self.conf = conf
        self.num_nodes = self.get_dataset_size()
        self.item_source_id_mappings = self.get_item_source_mappings()
        self.bidirectional = conf['bidirectional_graph']
        self.split_data = conf['split_data']
        if "graph_batching" not in conf.keys():
            self.graph_batching = False
        else:
            self.graph_batching = conf['graph_batching']

        # load train/val/test data seperately
        if self.split_data:
            self.train_pairs_data, self.adj_graph_train = self.get_graph("item_transition_train.csv", self.bidirectional)
            self.val_pair_data, self.adj_graph_val = self.get_graph("item_transition_val.csv", self.bidirectional)
            self.test_pair_data, self.adj_graph_test = self.get_graph("item_transition_test.csv", self.bidirectional)

            print_statistics(self.adj_graph_train, "training set")
            print_statistics(self.adj_graph_val, "validation set")
            print_statistics(self.adj_graph_test, "testing set")

            if self.graph_batching:
                self.train_set = TrainDataGraphBatching(self.train_pairs_data, self.adj_graph_train, conf['num_sampling_neighbors'], conf['num_layers'])
                self.val_set = TestData(self.val_pair_data, self.adj_graph_val, self.adj_graph_train)
                self.test_set = TestData(self.test_pair_data, self.adj_graph_test, self.adj_graph_train)

                self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=30, drop_last=True, collate_fn=self.train_set.collate_sampled_sub_graphs)
                self.test_loader = DataLoader(self.test_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=8)
                self.val_loader = DataLoader(self.val_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=8)
            
            else:
                self.train_set = TrainData(self.train_pairs_data, self.adj_graph_train)
                self.val_set = TestData(self.val_pair_data, self.adj_graph_val, self.adj_graph_train)
                self.test_set = TestData(self.test_pair_data, self.adj_graph_test, self.adj_graph_train)

                self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10, drop_last=True)
                self.test_loader = DataLoader(self.test_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)
                self.val_loader = DataLoader(self.val_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)

        # load the entire dataset for pretraining
        else:
            self.train_pairs_data, self.adj_graph_train = self.get_graph("item_transition_pairs.csv", self.bidirectional)
            print_statistics(self.adj_graph_train, "training set")

            if self.graph_batching:
                self.train_set = TrainDataGraphBatching(self.train_pairs_data, self.adj_graph_train, conf['num_sampling_neighbors'], conf['num_layers'])
                self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=30, drop_last=True)

            else:
                self.train_set = TrainData(self.train_pairs_data, self.adj_graph_train)
                self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=20, drop_last=True)


    def get_dataset_size(self):
        data_path = self.conf["data_path"]        
        with open(data_path + "item_source_id_mapping.json", 'r') as f:
            load_obj = json.load(f)

        num_nodes = len(load_obj.keys())
        return num_nodes


    def get_item_source_mappings(self):
        data_path = self.conf["data_path"]        
        with open(data_path + "item_source_id_mapping.json", 'r') as f:
            raw_mappings = json.load(f)
        
        mappings = {}
        for k, v in raw_mappings.items():
            mappings[int(k)] = v

        return mappings


    def get_graph(self, filename, bi_directional_graph=True):
        data_path = self.conf["data_path"]
        edge_pairs = pd.read_csv(data_path + filename, sep=',', header=None).values

        if bi_directional_graph:
            reverse_pairs = np.stack([edge_pairs[:, 1], edge_pairs[:, 0]], axis=1)
            edge_pairs = np.concatenate([edge_pairs, reverse_pairs])

        indice = np.array(edge_pairs, dtype=np.int32)
        values = np.ones(len(edge_pairs), dtype=np.float32)
        adj_graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.num_nodes, self.num_nodes)).tocsr()

        return edge_pairs, adj_graph
    

    def yeild_sampled_full_graph(self):
        assert self.graph_batching, "Error - function yeild_sampled_full_graph is vaild only for dataloaders with graph sampling."
        return self.train_set.yeild_sampled_full_graph()