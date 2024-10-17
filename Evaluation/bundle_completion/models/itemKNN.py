import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class itemKNN(nn.Module):
    def __init__(self, conf, pretrained_embeddings=None):
        super(itemKNN, self).__init__()
        self.conf = conf
        self.n_items = conf["num_items"]
        self.emb_size = conf['emb_size']
        self.device = conf["device"]

        if self.emb_size is None:
            assert pretrained_embeddings is not None, "Failed to initialize itemKNN, an input pretrained embedding is necessary if emb_size is set to None."
            self.emb_size = pretrained_embeddings.shape[1]

        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape[1] == self.emb_size, "Failed to initialize itemKNN, embedding size does not match with the pre-trained embeddings."

        self.items_embedding = self.init_embeddings(pretrained_embeddings)


    def init_embeddings(self, pretrained_embeddings=None):
        if pretrained_embeddings is None:
            initializer = nn.init.xavier_uniform_
            items_embedding = initializer(torch.empty(self.n_items, self.emb_size)).to(self.device)
            items_embedding = Parameter(items_embedding, requires_grad=True)
        else:
            items_embedding = Parameter(pretrained_embeddings, requires_grad=True)
        return items_embedding
    

    def init_metadata(self, metadata):
        self.item_metadata = np.array(metadata)

    
    # Input a batch of item_ids (np.ndarray # [batch_size])
    # Returns the corresponding item metadata
    def get_item_metadata(self, item_ids):
        item_metadata = self.item_metadata[item_ids]
        return item_metadata


    def evaluate_similarity_scores(self, items_id_mask):
        # shape of items_id_mask - [batch_size, num_items]
        # shape of items_center_rep - [batch_size, emb_size]
        items_center_sum = torch.mm(items_id_mask, self.items_embedding)
        items_num = torch.sum(items_id_mask, dim=1).unsqueeze(dim=1).expand(-1, self.emb_size)
        items_center_rep = items_center_sum / items_num

        scores = torch.mm(items_center_rep, self.items_embedding.t())
        return scores


    def forward(self, items_id_mask):
        similarity_scores = self.evaluate_similarity_scores(items_id_mask)

        return similarity_scores
