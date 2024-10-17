import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from Evaluation.bundle_completion.models.itemKNN import itemKNN
from Evaluation.bundle_completion.utils.pretrained_multimodal_embedding_utils import map_item_meta_features


def evaluate(conf, dataset, item_meta_features, pretrained_feature_mappings):
    log_path = "./Evaluation/bundle_completion/log/%s/%s" % (conf["dataset"], conf["model"])
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    settings = []
    if conf["info"] != "":
        settings.append(conf["info"])
    settings += [conf['pretrain_alias']]
    if 'ckpt_file_name' in conf.keys():
        settings += [conf['ckpt_file_name']]
    if 'feature_extraction_type' in conf.keys():
        settings += [conf['feature_extraction_type']]
    setting = "_".join(settings)

    log_path = log_path + "/" + setting
    device = conf['device']

    # model
    if conf['model'] == 'itemKNN':
        dataset_mapped_id_to_source_mappings = dataset.get_item_id_to_source_mappings()
        pretrained_embedding = map_item_meta_features(item_meta_features, pretrained_feature_mappings, dataset_mapped_id_to_source_mappings, device)
        model = itemKNN(conf, pretrained_embedding).to(device)
    else:
        raise ValueError("Unimplemented model %s" %(conf["model"]))

    epoch = 1
    batch_anchor = 0
    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0

    # For ItemKNN, no fine-tuning is applied. We directly evaluate the item similarity based on pre-trained embeddings
    metrics = {}
    metrics["val"] = test(model, dataset.val_loader, conf)
    metrics["test"] = test(model, dataset.test_loader, conf)
    best_metrics, best_perform, best_epoch = log_metrics(conf, metrics, log_path, epoch, batch_anchor,
                                                         best_metrics, best_perform, best_epoch)
    return best_metrics
    

# $pretrained_feature_mappings: type - np.array of shape [N, 2].
#   - Each row contains item idx corresponding to the input feature tensor and the source id.
def k_cross_evaluate(conf, dataset, item_meta_features, pretrained_feature_mappings, validation_k, metrics):
    device = conf['device']
    # model
    if conf['model'] == 'itemKNN':
        dataset_mapped_id_to_source_mappings = dataset.mapped_to_sid_mappings
        pretrained_embedding = map_item_meta_features(item_meta_features, pretrained_feature_mappings, dataset_mapped_id_to_source_mappings, device)
        model = itemKNN(conf, pretrained_embedding).to(device)
    else:
        raise ValueError("Unimplemented model %s" %(conf["model"]))

    # For ItemKNN, no fine-tuning is applied. We directly evaluate the item similarity based on pre-trained embeddings
    metrics[validation_k]["val"] = test(model, dataset.val_loader, conf)
    metrics[validation_k]["test"] = test(model, dataset.test_loader, conf)
    

def log_k_cross_mertics(conf, metrics):
    merged_metrics = get_k_cross_metrics(metrics, conf['topk'], conf['k_cross'])

    log_path = "./Evaluation/bundle_completion/results/log/%s_cross_%d/%s" % (conf["dataset"], conf['k_cross'], conf["model"])
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    settings = []
    if conf["info"] != "":
        settings.append(conf["info"])
    if 'pretrain_alias' in conf.keys():
        settings += [conf['pretrain_alias']]
    if 'ckpt_file_name' in conf.keys():
        settings += [conf['ckpt_file_name']]
    if 'feature_extraction_type' in conf.keys():
        settings += [conf['feature_extraction_type']]
    setting = "_".join(settings)

    log_path = log_path + "/" + setting

    best_metrics, best_perform = init_best_metrics(conf)
    epoch = 1
    batch_anchor = 0
    best_epoch = 0
    best_metrics, best_perform, best_epoch = log_metrics(conf, merged_metrics, log_path, epoch, batch_anchor,
                                                        best_metrics, best_perform, best_epoch)

    return best_metrics



def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key in best_metrics:
            for metric in best_metrics[key]:
                best_metrics[key][metric][topk] = -1
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def get_k_cross_metrics(total_metrics, topks, num_validation_groups):
    metrics = {
        'val': {
            'recall': {},
            'recall_range': {},
            'recall_std': {},
            'ndcg': {},
            'ndcg_range': {},
            'ndcg_std': {},
        },
        'test': {
            'recall': {},
            'recall_range': {},
            'recall_std': {},
            'ndcg': {},
            'ndcg_range': {},
            'ndcg_std': {},
        },
    }

    dataset_types = ['val', 'test']
    evaluations = ['recall', 'ndcg']

    for dataset_type in dataset_types:
        for evaluation in evaluations:
            for topk in topks:
                eval_vals = []

                for group_i in range(num_validation_groups):
                    eval_vals.append(total_metrics[group_i][dataset_type][evaluation][topk])
                
                eval_vals = np.array(eval_vals)
                metrics[dataset_type][evaluation][topk] = eval_vals.mean()
                metrics[dataset_type][evaluation + "_range"][topk] = (eval_vals.max() - eval_vals.min()) / 2
                metrics[dataset_type][evaluation + "_std"][topk] = eval_vals.std()

    return metrics


def write_log(log_path, topk, step, metrics):
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


def log_metrics(conf, metrics, log_path, epoch, batch_anchor, best_metrics, best_perform,
                best_epoch):
    for topk in conf["topk"]:
        write_log(log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 10
    print("top%d as the final evaluation standard" % (topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > \
            best_metrics["val"]["ndcg"][topk_]:
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            # print(best_perform["val"][topk])
            # print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()

    with torch.no_grad():

        for batch_cnt, batch in enumerate(dataloader):
            batch = [i.to(conf['device']) for i in batch]

            incomplete_item_set, target_id = batch

            scores = model(incomplete_item_set).detach()  # [bs, num_items]
            scores -= 1e8 * incomplete_item_set
            tmp_metrics = get_metrics(tmp_metrics, target_id, scores, conf["topk"])

        torch.cuda.empty_cache()

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}

    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                 dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit.to(device)
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float, device=device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)
    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]
