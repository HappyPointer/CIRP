import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import time
from pathlib import Path
from itertools import product
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import BLIP_pretraining.models.blip_consistent_CF_finetune as BLIP_consistent_CF_Finetune
import BLIP_pretraining.utils.utils as utils
from BLIP_pretraining.utils.utils import warmup_lr_schedule, step_lr_schedule
from BLIP_pretraining.data.relational_data_utils import create_dataset, create_sampler, create_loader


# Model training with consistent strategy, designed for training BLIP_consistent_CF_Finetune.BLIP
def consistent_train(model, data_loader, optimizer, epoch, device, summary_writer, config):
    # train
    train_start_time = time.time()

    model.train()

    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('ego_itc_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('cross_itc_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    
    data_loader.sampler.set_epoch(epoch)
    
    epoch_batch_offset = len(data_loader) * epoch
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        ego_image_text_pairs, cross_image_text_pairs = batch_data
        _, ego_image, ego_caption = ego_image_text_pairs
        _, cross_image, cross_caption = cross_image_text_pairs

        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
        
        optimizer.zero_grad()
        
        ego_image = torch.stack(ego_image, dim=0)
        ego_image = ego_image.to(device,non_blocking=True)
        cross_image = torch.stack(cross_image, dim=0)
        cross_image = cross_image.to(device,non_blocking=True)
        
        # ramp up alpha in the first 2 epochs
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader))) 

        losses = BLIP_consistent_CF_Finetune.forward_and_backward(model, (ego_image, ego_caption), (cross_image, cross_caption), alpha = alpha)
        # backward operation has been completed in forward_and_backward function
        optimizer.step()  

        loss = losses['ego_itc'] + losses['cross_itc']

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(ego_itc_loss=losses['ego_itc'].item())
        metric_logger.update(cross_itc_loss=losses['cross_itc'].item())

        batch_i = epoch_batch_offset + i
        for k, meter in metric_logger.meters.items():
            summary_writer.add_scalar(k, meter.global_avg, batch_i)

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    train_states = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  
    # time are recorded in minute format
    train_states['time'] = (time.time() - train_start_time) / 60
    
    return train_states


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset(config, min_scale=0.2)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[datasets[0].collate_relational_batch])[0] 
    # batch output: (anchor_captions, anchor_images), (pos_captions, pos_images), (neg_captions, neg_images)

    #### Model #### 
    print("Creating model")
    model = BLIP_consistent_CF_Finetune.blip_finetune(med_config='BLIP_pretraining/configs/bert_config.json', image_size=config['image_size'], 
                            vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
    
    model = model.to(device)   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    if config['checkpoint']:    
        checkpoint = torch.load(config['checkpoint'], map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)
        model.fix_first_half_encoders()
        
        if config['load_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']+1                
            print('resume checkpoint from %s'%config['checkpoint'])    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    

    summary_writer = SummaryWriter(os.path.join(args.output_dir, "training_curves/"))

    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        train_stats = consistent_train(model, data_loader, optimizer, epoch, device, summary_writer, config)
        
        if utils.is_main_process(): 
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }  
            if epoch % config['save_ckpt_interval'] == 0:                    
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                  
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./BLIP_pretraining/configs/relational_pretrain.yaml')
    parser.add_argument('--output_info', default='test_run')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    for CF_dataset_path, checkpoint_and_alias in product(config['CF_dataset_paths'], config['checkpoints_and_aliases']):
        config['CF_dataset_path'] = CF_dataset_path
        config['checkpoint'] = checkpoint_and_alias[0]
        alias = checkpoint_and_alias[1]

        output_dir = f"output/{config['dataset_name']}/{args.output_info}_ckpt_{alias}_" +\
            f"Args_lr_{config['init_lr']}_WD{config['weight_decay']}_bs_{config['batch_size']}_loss"

        args.output_dir = output_dir

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
        main(args, config)