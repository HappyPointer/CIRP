import os
import torch
from itertools import product
import yaml
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from item_transition_modeling.LightGCN import LightGCN
from item_transition_modeling.utils import Datasets


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="clothing", type=str, help="dataset, options: clothing, electronic, food")
    parser.add_argument("-m", "--model", default="LightGCN", type=str, help="which model to use, options: LightGCN")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-sp", "--split", default=0, type=int, help="when setting to 0, the entire dataset will used for pre-training, set to 1 for tuning hyper-parameters.")
    args = parser.parse_args()
    return args


def main():
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    model_name = paras["model"]
    assert model_name in ["LightGCN"], "Unknown model name {}".format(model_name)

    if model_name == "LightGCN":
        conf = yaml.safe_load(open("./item_transition_modeling/transition_LightGCN_config.yaml"))
    print("Load config done!")

    dataset_name = paras["dataset"]

    conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["model"] = paras["model"]
    conf["split_data"] = bool(paras["split"])

    dataset = Datasets(conf)
    conf['num_nodes'] = dataset.num_nodes

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    for lr, aug_rate, emb_size, l2_reg, num_layers in \
            product(conf["lrs"], conf["aug_rates"], conf["emb_sizes"], conf["l2_regs"], conf['num_layerss']):
        
        if conf['split_data']:
            log_path = "./item_transition_modeling/outputs/log/%s/%s" % (conf["dataset"], conf["model"])
            run_path = "./item_transition_modeling/outputs/runs/%s/%s" % (conf["dataset"], conf["model"])
            checkpoint_model_path = "./item_transition_modeling/outputs/checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
            checkpoint_conf_path = "./item_transition_modeling/outputs/checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])        
        else:
            log_path = "./item_transition_modeling/pretrained/log/%s/%s" % (conf["dataset"], conf["model"])
            run_path = "./item_transition_modeling/pretrained/runs/%s/%s" % (conf["dataset"], conf["model"])
            checkpoint_model_path = "./item_transition_modeling/pretrained/checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
            checkpoint_conf_path = "./item_transition_modeling/pretrained/checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])              
        
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        settings = []
        if conf["info"] != "":
            settings.append(conf["info"])

        conf["lr"] = lr
        conf["aug_rate"] = aug_rate
        conf["emb_size"] = emb_size
        conf["l2_reg"] = l2_reg
        conf['num_layers'] = num_layers
        settings += ["LR" + str(lr), str(conf['aug_type']) + str(aug_rate), "emb" + str(emb_size),
                     "WD" + str(l2_reg), "layer" + str(num_layers)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)

        # model
        if conf['model'] == 'LightGCN':
            model = LightGCN(conf, dataset.adj_graph_train).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))

        print(conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])
        print("%s start training ... " % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf['ed_interval'])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf["num_epoches"]):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
            
            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(conf["device"]) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf['aug_type'] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                    ED_drop = True

                loss = model(batch, ED_drop)
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                run.add_scalar("loss", loss_scalar, batch_anchor)
                pbar.set_description("epoch: %d, loss: %.4f" % (epoch, loss_scalar))

                if conf['split_data'] and (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)

                        
        if not conf['split_data']:
            # if we use the entire dataset for pre-training with evaluation, save the model after training completed.
            save_model(conf, model, checkpoint_model_path, checkpoint_conf_path)


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["hit_rate"] = {}
    for topk in conf['topk']:
        for key in best_metrics:
            for metric in best_metrics[key]:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" % (m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, hit_rate: %f" % (
    curr_time, topk, val_scores["recall"][topk], val_scores["hit_rate"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, hit_rate: %f" % (
    curr_time, topk, test_scores["recall"][topk], test_scores["hit_rate"][topk])

    log = open(log_path, "a")
    log.write("%s\n" % (val_str))
    log.write("%s\n" % (test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform,
                best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 10
    print("top%d as the final evaluation standard" % (topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["hit_rate"][topk_] > \
            best_metrics["val"]["hit_rate"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))

        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, hit_rate_T=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["hit_rate"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, hit_rate_V=%.5f" % (
            curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["hit_rate"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch
    


def save_model(conf, model, checkpoint_model_path, checkpoint_conf_path):
    torch.save(model.state_dict(), checkpoint_model_path)
    dump_conf = dict(conf)
    del dump_conf["device"]
    json.dump(dump_conf, open(checkpoint_conf_path, "w"))
    print(f"The trained model has been successfully saved in {checkpoint_model_path}.")


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "hit_rate"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    results = model.propagate(test=True)
    with torch.no_grad():
        for node_ids, ground_truth, train_mask in dataloader:
            ground_truth = ground_truth.to(device)
            pred = model.evaluate(results, node_ids.to(device))
            pred -= 1e8 * train_mask.to(device)
            tmp_metrics = get_metrics(tmp_metrics, ground_truth, pred, conf["topk"])

        torch.cuda.empty_cache()

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "hit_rate": {}}

    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                 dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["hit_rate"][topk] = get_hit_rate(grd, is_hit)

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


def get_hit_rate(grd, is_hit):
    hit_cnt = is_hit.sum().item()
    num_pos = grd.sum().item()
    return [hit_cnt, num_pos]


if __name__ == "__main__":
    main()