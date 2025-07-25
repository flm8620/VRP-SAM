r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)
        protos, _ = model(args.condition, batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training)

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        # print(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=200)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss', 'lvis', 'paco_part', 'fss', 'pascal_part'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--sam_version', type=str, default='vit_h')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    # Distributed setting
    local_rank = args.local_rank
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    #device = torch.device('cuda', local_rank)
    device = torch.device('cuda')
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    start_epoch = 0
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    
    if args.resume is not None:
        if utils.is_main_process():
            Logger.info(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if 'model_state_dict' in checkpoint:
            # New checkpoint format
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(model_state_dict)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_miou = checkpoint.get('val_miou', float('-inf'))
            best_val_loss = checkpoint.get('val_loss', float('inf'))
        else:
            # Old format - just state dict
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(model_state_dict)   

    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred(args.sam_version)
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.transformer_decoder.parameters()},
        {'params': model.module.downsample_query.parameters(), "lr": args.lr},
        {'params': model.module.merge_1.parameters(), "lr": args.lr},
        
        ],lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    
    # Resume optimizer and scheduler if available
    if args.resume is not None and 'optimizer_state_dict' in torch.load(args.resume, map_location='cpu'):
        checkpoint = torch.load(args.resume, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if utils.is_main_process():
            Logger.info('Resumed optimizer and scheduler states')

    # Training 
    for epoch in range(start_epoch, args.epochs):

        trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)

        # Save checkpoint for every epoch and best model
        if utils.is_main_process():
            # Save checkpoint every epoch
            Logger.save_model_checkpoint(model, optimizer, scheduler, epoch, val_miou, val_loss)
            
            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            Logger.tbd_writer.add_scalar('train/loss', trn_loss, epoch)
            Logger.tbd_writer.add_scalar('val/loss', val_loss, epoch)
            Logger.tbd_writer.add_scalar('train/miou', trn_miou, epoch)
            Logger.tbd_writer.add_scalar('val/miou', val_miou, epoch)
            Logger.tbd_writer.add_scalar('train/fb_iou', trn_fb_iou, epoch)
            Logger.tbd_writer.add_scalar('val/fb_iou', val_fb_iou, epoch)
            Logger.tbd_writer.add_scalar('learning_rate', current_lr, epoch)
            Logger.tbd_writer.flush()
    
    if utils.is_main_process():
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')