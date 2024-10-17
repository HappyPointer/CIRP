import os
import torch
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from BLIP_pretraining.utils.feature_extraction import extract_feature
import Evaluation.bundle_completion.evaluation_itemKNN as bundle_completion_itemKNN_eval
import Evaluation.bundle_completion.utils.bundle_completion_datasets as bundle_completion_datasets


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="clothing", type=str, help="dataset, options: clothing, electronic, food")
    parser.add_argument("-m", "--model", default="itemKNN", type=str, help="which model to use")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-k", "--k_cross", default="5", type=str, help="total number of groups split for cross validation, set to 0 if cross-validation is not applied. Ignored if downstream evaluation does not support k-cross evaluation.")
    args = parser.parse_args()
    return args


def main():
    # Please change the config below to test the performance of different checkpoints.
    # pretrain_alias: alias of the evaluated pretraining checkpoint, used for naming the evaluation result
    # pretrain_ckpt_path: path of the checkpoint which need to be evluated
    conf = {
        "pretrain_alias": "BLIP_finetuned",
        "pretrain_ckpt_path": "./saved_checkpoints/BLIP_finetuned/checkpoint_09.pth",
        "feature_extraction_type": "ti_norm_mix",
    }

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    conf["dataset"] = dataset_name
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["model"] = paras["model"]
    conf['k_cross'] = int(paras['k_cross'])

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    # ===================================================
    # Model evaluation on bundle completion task

    # check if item meta-feature has been extracted and saved before, if not, extract the item meta-features
    extraction_base_directory = os.path.dirname(conf['pretrain_ckpt_path'])
    ckpt_file_name, _ = os.path.splitext(os.path.basename(conf['pretrain_ckpt_path']))
    conf['ckpt_file_name'] = ckpt_file_name
    meta_feature_directory = f"{extraction_base_directory}/bundle_completion/extraction_{ckpt_file_name}/{dataset_name}"

    extracted_feature_path = f"{meta_feature_directory}/{conf['feature_extraction_type']}_features.pkl"
    item_id_mapping_path = f"{meta_feature_directory}/{conf['feature_extraction_type']}_item_source_ids.csv"

    if conf['model'] == "itemKNN":
        task_config = yaml.safe_load(open("./Evaluation/bundle_completion/configs/config_itemKNN.yaml"))
        for k, v in task_config[dataset_name].items():
            conf[k] = v
    else:
        raise Exception(f"Unimplemented model {conf['model']} for Bundle Completion evaluation.")
    

    if os.path.exists(extracted_feature_path) and os.path.exists(item_id_mapping_path):
        print("Load local backup for extracted item meta features.")
        with open(extracted_feature_path, 'rb') as f:
            item_features = pickle.load(f)
        source_id_mappings = np.loadtxt(item_id_mapping_path, delimiter=',', dtype=str)

    else:
        sid_mapping_path = conf["data_path"] + "item_idx_mapping.csv"
        item_source_ids = pd.read_csv(sid_mapping_path, sep=',')["source ID"].tolist()
        print(f"Local extraction backup not found. Start extracting {dataset_name} item meta-features with {conf['pretrain_ckpt_path']} for Bundle Completion:")
        item_features, source_id_mappings = extract_feature(conf['pretrain_ckpt_path'], conf['feature_extraction_type'], item_source_ids, conf['metadata_path'], meta_feature_directory, device)

    # k-fold cross validation
    if conf['k_cross'] > 0:
        metrics = {}
        for i in range(conf['k_cross']):
            metrics[i] = {}

        for cross_idx_i in range(conf['k_cross']):
            dataset = bundle_completion_datasets.Dataset(conf, cross_validation=True, total_validation_groups=conf['k_cross'], cross_validation_idx=cross_idx_i)
            conf['num_bundles'] = dataset.n_bundles
            conf['num_items'] = dataset.n_items
            bundle_completion_itemKNN_eval.k_cross_evaluate(conf, dataset, item_features, source_id_mappings, cross_idx_i, metrics)

        eval_result = bundle_completion_itemKNN_eval.log_k_cross_mertics(conf, metrics)

    # normal evaluation process
    else:
        dataset = bundle_completion_datasets.Dataset(conf)
        conf['num_bundles'] = dataset.n_bundles
        conf['num_items'] = dataset.n_items
        eval_result = bundle_completion_itemKNN_eval.evaluate(conf, dataset, item_features, source_id_mappings)

    
    log_path = "./Evaluation/bundle_completion/perf_curves/%s/%s/" % (conf["dataset"], conf["model"])

    settings = []
    if conf["info"] != "":
        settings.append(conf["info"])
    settings += [conf['pretrain_alias'], conf['feature_extraction_type']]
    setting = "_".join(settings)

    log_path = log_path + "/" + setting + "/"
    run = SummaryWriter(log_path + "perf_curves")

    for topk in conf["topk"]:
        write_log(log_path + "perf_statistics.txt", topk, eval_result)


def write_log(log_path, topk, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
    curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
    curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" % (val_str))
    log.write("%s\n" % (test_str))
    log.close()

    print(val_str)
    print(test_str)


if __name__ == "__main__":
    main()