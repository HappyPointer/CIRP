import torch
import json
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
import pandas as pd

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from BLIP_pretraining.data.utils import pre_caption
from datasets.Amazon_metadata.amazon_metadata_utils import Amazon_metadata_loader


# Load relational pretrain data from the indicated path containing II relation pairs (id-based) and id-source mapping json.
# para - keep_ego_pairs: indicate whether the oringal image-text pairs (ego image-text pairs) of the items should be generated or not.
class relational_pretrain_dataset(Dataset):
    def __init__(self, dataset_path, metadata_path, transform): 
        self.dataset_path = dataset_path
        self.source_to_id_mappings, self.id_to_source_mappings = self.load_source_id_mappings()
        self.num_nodes = len(self.source_to_id_mappings.keys())
        self.adj_pairs, self.adj_graph = self.load_dataset()
        self.num_pairs = self.adj_pairs.shape[0]

        self.transform = transform
        self.metadata_loader = Amazon_metadata_loader(metadata_path)


    def load_dataset(self):
        edge_pairs = pd.read_csv(self.dataset_path + "item_transition_pairs.csv", sep=',', header=None).values        
        indice = np.array(edge_pairs, dtype=np.int32)
        values = np.ones(len(edge_pairs), dtype=np.float32)
        adj_graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.num_nodes, self.num_nodes)).tocsr()
        return edge_pairs, adj_graph
    

    def load_source_id_mappings(self):
        mapping_path = self.dataset_path + 'item_source_id_mapping.json'
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


    def __len__(self):
        return self.num_pairs
    

    def __getitem__(self, index):
        anchor, pos_end = self.adj_pairs[index]
        return int(anchor), int(pos_end)
    

    # Given a batch input where items are represented by item ids, this function tranforms item ids into item caption and image objects.
    # Items without valid images will be kept with image field left as None.
    # para: $batch_ids [tuple of ints] - Batch item ids to be processed.
    def process_batch_item_ids(self, batch_ids):
        batch_source_ids = [self.id_to_source_mappings[id] for id in batch_ids]
        raw_texts, image_paths = self.metadata_loader.retrieve_metadata(batch_source_ids, invalid_image_process='keep')

        captions = []
        images = []
        for i in range(len(raw_texts)):
            caption, image = self.process_image_text(raw_texts[i], image_paths[i])
            captions.append(caption)
            images.append(image)

        return images, captions


    def process_image_text(self, caption, image_path):
        if caption is not None:
            caption = pre_caption(caption, 50)

        if image_path is not None:
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)
        else:
            image = None
        
        return caption, image
    

    # For cross-item pairs, the diagonal of image-text similarity matrix indicates the cross-item matching pairs
    # The upper left / lower right half diagonal indicates the original image-text pair (ego item pair)
    # We need to mask both cross-item pairs and ego-item pairs.
    def generate_batch_cross_item_mask(self, num_items):
        mask = np.diag(np.ones(num_items * 2))
        mask += np.diag(np.ones(num_items), num_items)
        mask += np.diag(np.ones(num_items), -num_items)
        mask = 1 - mask
        return mask
    

    def collate_relational_batch(self, batch):
        (anchor_ids, pos_ids) = zip(*batch)
        
        anchor_images, anchor_captions = self.process_batch_item_ids(anchor_ids)
        pos_images, pos_captions = self.process_batch_item_ids(pos_ids)

        valid_item_pair_indices = []
        for i in range(len(anchor_ids)):
            if (anchor_images[i] is not None) and (pos_images[i] is not None):
                valid_item_pair_indices.append(i)

        # remove item pairs with invalid images
        iter_lists = [anchor_ids, anchor_images, anchor_captions,
                      pos_ids, pos_images, pos_captions]

        for i, list_obj in enumerate(iter_lists):
            iter_lists[i] = [item for idx, item in enumerate(list_obj) if idx in valid_item_pair_indices]

        # return (anchor_ids, anchor_images, anchor_captions), (pos_ids, pos_images, pos_captions)
        return (iter_lists[0], iter_lists[1], iter_lists[2]), (iter_lists[3], iter_lists[4], iter_lists[5])
