import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
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


# given a csr_matrix, remove rows where all elements in the row are zeros
def remove_zero_rows(input_matrix):
    nonzero_row_indice, _ = input_matrix.nonzero()
    nonzero_row_indice = np.unique(nonzero_row_indice)
    return input_matrix[nonzero_row_indice]    


def pad_collate(batch):
    (batch_idx, batch_list) = zip(*batch)
    batch_list = torch.stack(batch_list)
    
    len_list = [len(x) for x in batch_idx]
    batch_idx_pad = pad_sequence(batch_idx, batch_first=True, padding_value=-1)
    return batch_idx_pad, len_list, batch_list



class TrainData(Dataset):
    def __init__(self, conf, bi_pairs, bi_graph_train, seq_format):
        self.conf = conf
        self.bi_pairs = bi_pairs
        self.bi_graph = remove_zero_rows(bi_graph_train)
        self.n_items = bi_graph_train.shape[1]
        self.seq_format = seq_format
        if seq_format:
            self.largest_bundle_size = self.get_largest_bundle_size()

    def __len__(self):
        return self.bi_graph.shape[0]
    
    def get_largest_bundle_size(self):
        return int(self.bi_graph.toarray().sum(axis=1).max())

    def __getitem__(self, idx):
        items = self.bi_graph[idx].toarray().squeeze()
        if self.seq_format:
            items_idx = self.bi_graph[idx].indices
            # items_idx = np.zeros(self.largest_bundle_size) - 1
            # items_idx[:len(item_contents)] = item_contents 
            return torch.from_numpy(items_idx), torch.from_numpy(items)
        
        else:
            return torch.from_numpy(items)
        


class TestData(Dataset):
    def __init__(self, conf, bi_graph, bi_pairs, n_items):
        self.conf = conf
        self.bi_graph = bi_graph
        self.bi_pairs = bi_pairs
        self.n_items = n_items

    def __len__(self):
        return len(self.bi_pairs)

    def __getitem__(self, idx):
        # the bi_pair to be discarded as the completion target
        bi_pair = self.bi_pairs[idx]

        b_id = bi_pair[0]
        i_id = bi_pair[1]

        aff_items = self.bi_graph[b_id].toarray().squeeze()
        assert aff_items[i_id] == 1, f"Error: sampled item {i_id} does not exist in bundle {b_id}."
        target_item = np.zeros(self.n_items)
        target_item[i_id] = 1

        incomplete_item_set = aff_items.copy()
        incomplete_item_set[i_id] = 0

        return incomplete_item_set, target_item


class Dataset():
    def __init__(self, conf, seq_format=False, cross_validation=False, total_validation_groups=0, cross_validation_idx=0, show_statistics=False):
        self.conf = conf
        self.n_bundles, self.n_items = self.get_dataset_size()
        # if seq_format is set to True, the training data will return indices of non-zero items together with one-hot item list.
        self.seq_format = seq_format
        self.ori_to_mapped_idx_mapping, self.mapped_to_ori_idx_mapping = self.generate_item_mapping()
        self.mapped_to_sid_mappings, self.sid_to_mapped_mappings = self.build_meta_source_id_mappings(self.ori_to_mapped_idx_mapping)

        if not cross_validation:
            self.bi_pairs_train, self.bi_graph_train = self.get_graph("bundle_item_train.csv")
            self.bi_pairs_val, self.bi_graph_val = self.get_graph("bundle_item_val.csv")
            self.bi_pairs_test, self.bi_graph_test = self.get_graph("bundle_item_test.csv")
        else:
            assert total_validation_groups > 0 and cross_validation_idx < total_validation_groups, \
            "Error. Please input the correct number of cross validation groups and current validation index."
            self.bi_pairs_train, self.bi_graph_train, \
            self.bi_pairs_val, self.bi_graph_val, \
            self.bi_pairs_test, self.bi_graph_test = \
                self.get_graph_under_cross_validation(cross_validation_idx, total_validation_groups)

        if show_statistics:
            print_statistics(self.bi_graph_train, "BI training set")
            print_statistics(self.bi_graph_val, "BI validation set")
            print_statistics(self.bi_graph_test, "BI testing set")
        
        self.train_set = TrainData(conf, self.bi_pairs_train, self.bi_graph_train, seq_format)
        if seq_format:
            self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10, collate_fn=pad_collate)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=10)

        self.val_set = TestData(conf, self.bi_graph_val, self.bi_pairs_val, self.n_items)
        self.val_loader = DataLoader(self.val_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)
        self.test_set = TestData(conf, self.bi_graph_test, self.bi_pairs_test, self.n_items)
        self.test_loader = DataLoader(self.test_set, batch_size=conf["test_batch_size"], shuffle=False, num_workers=10)


    def get_dataset_size(self):
        data_path = self.conf["data_path"]
        with open(data_path + "bundle_item_dataset_size.txt") as f:
            dataset_size_list = list(f.readlines())
            n_users = dataset_size_list[0].strip()
            n_items = dataset_size_list[1].strip()

        return int(n_users), int(n_items)


    def get_item_id_to_source_mappings(self):
        return self.mapped_to_sid_mappings


    # The item indexes in bundle_item.csv do not start from zero
    # The function rearrange the item indexes and return the item index mapping.
    def generate_item_mapping(self):
        data_path = self.conf["data_path"]
        # Note: Header exists in original bundle_item.csv. But the split datasets (train/val/test) do not contain a header
        bi_pairs = pd.read_csv(data_path + "bundle_item.csv", sep=',', header=None).values
        item_idx_list = sorted(list(set(bi_pairs[:, 1])))

        source_to_current_idx_mapping = {k: v for k, v in zip(item_idx_list, list(range(len(item_idx_list))))}
        current_to_source_idx_mapping = {v: k for k, v in zip(item_idx_list, list(range(len(item_idx_list))))}
        return source_to_current_idx_mapping, current_to_source_idx_mapping


    def get_graph(self, filename):
        data_path = self.conf["data_path"]
        bi_pairs_ori = pd.read_csv(data_path + filename, sep=',', header=None).values

        bi_pairs = self.generate_mapped_pairs(bi_pairs_ori)

        indice = np.array(bi_pairs, dtype=np.int32)
        values = np.ones(len(bi_pairs), dtype=np.float32)
        bi_graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.n_bundles, self.n_items)).tocsr()
        return bi_pairs, bi_graph


    def get_graph_under_cross_validation(self, test_group_idx, num_validation_groups):
        base_data_path = self.conf["data_path"] + "cross_validation_set/"
        # Load the groups of items for training
        training_group_indices = [i for i in range(num_validation_groups)]
        training_group_indices.remove(test_group_idx)

        bi_pairs_train = []
        for group_idx in training_group_indices:
            group_i_data_path = base_data_path + "bundle_item_{}.csv".format(group_idx)
            group_bi_pairs_ori = pd.read_csv(group_i_data_path, sep=',', header=None).values
            group_bi_pairs = self.generate_mapped_pairs(group_bi_pairs_ori)
            bi_pairs_train.append(group_bi_pairs)

        bi_pairs_train = np.vstack(bi_pairs_train)
        indice = np.array(bi_pairs_train, dtype=np.int32)
        values = np.ones(len(bi_pairs_train), dtype=np.float32)
        bi_graph_train = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.n_bundles, self.n_items)).tocsr()

        # Load the group of items for test and validation
        test_data_path = base_data_path + "bundle_item_{}.csv".format(test_group_idx)
        test_bi_pairs_ori = pd.read_csv(test_data_path, sep=',', header=None).values
        bi_pairs_test_and_val = self.generate_mapped_pairs(test_bi_pairs_ori)

        split_idx = int(bi_pairs_test_and_val.shape[0] / 2)
        bi_pairs_val = bi_pairs_test_and_val[:split_idx]
        bi_pairs_test = bi_pairs_test_and_val[split_idx:]

        indice = np.array(bi_pairs_val, dtype=np.int32)
        values = np.ones(len(bi_pairs_val), dtype=np.float32)
        bi_graph_val = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.n_bundles, self.n_items)).tocsr()

        indice = np.array(bi_pairs_test, dtype=np.int32)
        values = np.ones(len(bi_pairs_test), dtype=np.float32)
        bi_graph_test = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.n_bundles, self.n_items)).tocsr()

        return bi_pairs_train, bi_graph_train, bi_pairs_val, bi_graph_val, bi_pairs_test, bi_graph_test


    def generate_mapped_pairs(self, bi_pairs_ori):
        mapping = lambda x: self.ori_to_mapped_idx_mapping[x]
        bi_pairs_i = np.array(list(map(mapping, bi_pairs_ori[:, 1])))
        bi_pairs = np.column_stack((bi_pairs_ori[:, 0], bi_pairs_i))
        return bi_pairs


    # Build mappings between mapped item ids and source item ids used for querying Amazon metadata.
    def build_meta_source_id_mappings(self, ori_to_mapped_idx_mapping):
        sid_mapping_path = self.conf["data_path"] + "item_idx_mapping.csv"
        item_meta_ids = pd.read_csv(sid_mapping_path, sep=',')
        item_meta_ids['item ID'] = item_meta_ids['item ID'].map(ori_to_mapped_idx_mapping)

        mapped_to_sid_mappings = pd.Series(item_meta_ids['source ID'].values, index=item_meta_ids['item ID']).to_dict()
        sid_to_mapped_mappings = pd.Series(item_meta_ids['item ID'].values, index=item_meta_ids['source ID']).to_dict()
        return mapped_to_sid_mappings, sid_to_mapped_mappings
