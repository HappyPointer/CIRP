import numpy as np
import torch
import pickle
import json
import torch.nn as nn 


# Unfound item features are randomly initialized.
def map_item_meta_features(feature_embedding, pretrained_feature_mappings, dataset_mapped_id_to_source_mappings, device, print_statistics=False):
    item_embedding_shape = (len(dataset_mapped_id_to_source_mappings.keys()), feature_embedding.shape[1])
    item_feature_embeddings = torch.empty(item_embedding_shape)
    nn.init.xavier_normal_(item_feature_embeddings)

    # build the mapping from source ID to pretrained item features
    source_to_pretrain_feature_idx_mappings = {}
    for row in pretrained_feature_mappings:
        source_to_pretrain_feature_idx_mappings[row[1]] = int(row[0])

    miss_match_count = item_feature_embeddings.shape[0]

    for item_id in dataset_mapped_id_to_source_mappings.keys():
        source_id = dataset_mapped_id_to_source_mappings[item_id]
        if source_id in source_to_pretrain_feature_idx_mappings.keys():
            pretrain_feature_idx = source_to_pretrain_feature_idx_mappings[source_id]
            item_feature_embeddings[item_id] = feature_embedding[pretrain_feature_idx]

            miss_match_count -= 1
    
    if print_statistics:
        print(f"Item metadata has been generated. {miss_match_count}/{item_embedding_shape[0]} records are missing pretrianed embeddings after matching.")

    return item_feature_embeddings.to(device)


# metadata_path: path to meta json data with format [{item ID: val_a, image: val_b, caption: val_c}]
# An bug will be triggered if certain items do not find the corresponding metadata.
def map_item_meta_data(metadata_path, current_to_source_id_mapping):
    with open(metadata_path) as json_file:
        source_metadata = json.load(json_file)

    source_metadata_dict = {}
    for i in range(len(source_metadata)):
        metadata_i = source_metadata[i]
        source_metadata_dict[metadata_i['item ID']] = (metadata_i['image'], metadata_i['caption'])

    num_items = len(current_to_source_id_mapping.keys())
    metadata = [source_metadata_dict[current_to_source_id_mapping[i]] for i in range(num_items)]
    return metadata


# The input bi_graph should be a sparse csr_matrix.
# Given the mapped item embeddings, the function generates the corresponding pre-trained bundle embeddings by average aggregation.
def generate_bundle_embeddings(conf, item_embeddings, bi_graph):
    graph = bi_graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    sp_bi_graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to(conf['device'])
    bundle_embeddings = torch.matmul(sp_bi_graph, item_embeddings)

    return bundle_embeddings


def generate_user_embeddings(conf, item_embeddings, ui_graph):
    graph = ui_graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    sp_ui_graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to(conf['device'])
    user_embeddings = torch.matmul(sp_ui_graph, item_embeddings)

    return user_embeddings