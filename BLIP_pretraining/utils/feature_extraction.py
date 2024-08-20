from PIL import Image
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import product
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import BLIP_pretraining.models.blip as BLIP_base
from datasets.Amazon_metadata.amazon_metadata_utils import Amazon_metadata_loader
# os.environ["OPENBLAS_NUM_THREADS"] = "2"


def load_image(image_path, image_size, device):
    image_obj = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image_obj = transform(image_obj).unsqueeze(0).to(device)   
    return image_obj


# return extracted features (tensor of shape [num_items, hidden]) and the corresponding source item ids.
# input $pretrained_model: str indicating the saved checkpoint or a BLIP model object
# input $keep_invalids: if set to True, invalid source_ids without any caption or image data will be kept with a random tensor.
def extract_feature(BLIP_ckpt_path, extract_mode, item_source_ids, metadata_path, save_directory, device, keep_invalids=False):
    default_config = {
        "image_size": 224,
        "vit": "base",
        "item_shape": 768,
        "projected_dim": 256,
    }

    if isinstance(device, str):
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    # init the inference BLIP model
    image_size = default_config['image_size']

    if extract_mode in ['image', 'text', 'multimodal', 'ti_mix', 'ti_norm_mix']:
        model = BLIP_base.blip_feature_extractor(pretrained=BLIP_ckpt_path, image_size=image_size, vit=default_config['vit'], med_config='./BLIP_pretraining/configs/med_config.json')
        default_shape = default_config['item_shape']
    else:
        raise Exception("Unknown extract_model: {}".format(extract_mode))

    model.eval()
    model = model.to(device)

    # retreive item metainfo using source ids.
    metadata_loader = Amazon_metadata_loader(metadata_path)
    texts, image_paths = metadata_loader.retrieve_metadata(item_source_ids, invalid_image_process='keep')

    item_features = torch.zeros(len(item_source_ids), default_shape)

    valid_item_indices = []
    valid_sids = []
    for idx in tqdm(range(len(item_source_ids))):
        caption = texts[idx]
        image_path = image_paths[idx]

        # items without metadata
        if caption is None and image_path is None and not keep_invalids:
            continue

        try:
            image = load_image(image_path, image_size, device)     
        except:
            image = None

        if extract_mode == "multimodal":
            if image is not None:
                feature = model(image, caption, mode='multimodal', device=device)[0,0]
            else:
                feature = torch.rand(default_shape)
        
        elif extract_mode == "image":
            if image is not None:
                feature = model(image, caption, mode='image', device=device)[0,0]
            else:
                feature = torch.rand(default_shape)
        
        elif extract_mode == "text":
            if caption is not None:
                feature = model(image, caption, mode='text', device=device)[0,0]
            else:
                feature = torch.rand(default_shape)

        elif extract_mode == "ti_mix":
            if caption is not None:
                text_feature = model(image, caption, mode='text', device=device)[0,0]
            else:
                text_feature = torch.rand(default_shape)

            if image is not None:
                image_feature = model(image, caption, mode='image', device=device)[0,0]
            else:
                image_feature = torch.zeros(default_shape)
            
            feature = text_feature + image_feature
        
        elif extract_mode == "ti_norm_mix":
            if caption is not None:
                text_feature = model(image, caption, mode='text', device=device)[0,0]
            else:
                text_feature = torch.rand(default_shape)
            
            if image is not None:
                image_feature = model(image, caption, mode='image', device=device)[0,0]
                feature = F.normalize(text_feature, p=2, dim=-1) + F.normalize(image_feature, p=2, dim=-1)
            else:
                feature = F.normalize(text_feature, p=2, dim=-1) * 2
        
        else:
            raise Exception("Unknown extract_model: {}".format(extract_mode))

        with torch.no_grad():
            item_features[idx] = feature
            valid_item_indices.append(idx)
            valid_sids.append(item_source_ids[idx])

    
    # remove items without valid metadata
    item_features = item_features[valid_item_indices]
    source_id_arr = np.column_stack((list(range(len(valid_sids))), valid_sids))

    # save the feature dict
    if save_directory is not None:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        feature_save_path = f"{save_directory}/{extract_mode}_features.pkl"
        with open(feature_save_path, 'wb') as f:
            pickle.dump(item_features, f)

        id_save_path = f"{save_directory}/{extract_mode}_item_source_ids.csv"
        np.savetxt(id_save_path, source_id_arr, delimiter=',', fmt='%s')

    return item_features, source_id_arr


