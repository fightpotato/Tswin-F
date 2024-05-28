# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.init as init
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint_ft,
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_best,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor
)

from drloc import cal_selfsupervised_loss

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--weights', type=str, default=r'D:\QQDownLoad\pre_efficientnetv2-s.pth',
    #                     help='initial weights path')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    ## Augmentation & regularization parameters
    # parser.add_argument("--no-aug",action='store_true',default=True,
    #                     help="Disable all training augmentation, override other train aug args")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--use_drloc", default=False,action='store_true', help="Use Dense Relative localization loss")
    parser.add_argument("--drloc_mode", type=str, default="l1", choices=["l1", "ce", "cbr"])
    # parser.add_argument("--use_drloc", default=False, help="Use Dense Relative localization loss")
    parser.add_argument("--lambda_drloc", type=float, default=0.5, help="weight of Dense Relative localization loss")
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--use_multiscale", action='store_true')
    parser.add_argument("--ape", action="store_true", help="using absolute position embedding")
    parser.add_argument("--rpe", action="store_false", help="using relative position embedding")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--ssl_warmup_epochs", type=int, default=30)
    parser.add_argument("--total_epochs", type=int, default=300)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def _weight_decay(init_weight, epoch, warmup_epochs=10, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight


def main():
    tb_write = SummaryWriter()
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.TRAIN.BASE_LR = 2e-3
    # config.TRAIN.WARMUP_LR = 1e-3
    # config.TRAIN.MIN_LR = 1e-5
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    # dataset_train, data_loader_train, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(model)

    weight_path = r"F:\plantvillage\VT-model\cswin_small_224.pth"
    # weights_dict = torch.load(weight_path, map_location=device)
    # load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    # weights_dict = load_checkpoint()
    # print(model.load_state_dict(load_weights_dict, strict=False))
    load_checkpoint_ft(weight_path, model, logger)#预训练权重
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[config.LOCAL_RANK],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # supervised criterion监督准则
    if config.AUG.MIXUP > 0.:  # 标签平滑：减少真实样本标签的累扁在计算损失函数时的权重，最终起到抑制过拟合的效果。
        # smoothing is handled with mixup label transform
        criterion_sup = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_sup = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_sup = torch.nn.CrossEntropyLoss()

    # self-supervised criterion自监督准则
    criterion_ssup = cal_selfsupervised_loss

    max_accuracy = 0.0
    save_path="./best-modelforall.pth"

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        # light
        # dataset_val.cache.reset()
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    init_lambda_drloc = 0.0
    for epoch in range(config.TRAIN.START_EPOCH, 301):  # config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if config.TRAIN.USE_DRLOC:
            init_lambda_drloc = _weight_decay(
                config.TRAIN.LAMBDA_DRLOC,
                epoch,
                config.TRAIN.SSL_WARMUP_EPOCHS,
                config.TRAIN.EPOCHS)

        # train_one_epoch(
        #     config,
        #     model,
        #     criterion_sup,
        #     criterion_ssup,
        #     data_loader_train,
        #     optimizer,
        #     epoch,
        #     mixup_fn,
        #     lr_scheduler,
        #     logger,
        #     init_lambda_drloc
        # )
        train_metrics = train_one_epoch(config,
                                        model,
                                        criterion_sup,
                                        criterion_ssup,
                                        data_loader_train,
                                        optimizer,
                                        epoch,
                                        mixup_fn,
                                        lr_scheduler,
                                        logger,
                                        init_lambda_drloc)
        abc = OrderedDict(epoch=epoch)
        abc.update([('train_' + k, v) for k, v in train_metrics.items()])

        # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #     save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        # this is the save for pth by the stpe of 5

        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        tags = ['val_acc1', 'val_loss', 'train_loss', 'learn_rate']
        tb_write.add_scalar(tags[0], acc1, epoch)
        tb_write.add_scalar(tags[1], loss, epoch)
        tb_write.add_scalar(tags[2], abc["train_loss"], epoch)
        # tb_write.add_scalar(tags[1],)
        tb_write.add_scalar(tags[3], optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if dist.get_rank() == 0 and acc1 > max_accuracy:
            # 这个代码是将整个model权重进行存储
            torch.save(model.state_dict(), save_path)

            save_checkpoint_best(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(
        config,
        model,
        criterion_sup,
        criterion_ssup,
        data_loader,
        optimizer,
        epoch,
        mixup_fn,
        lr_scheduler,
        logger,
        lambda_drloc
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    device = torch.device('cuda', dist.get_rank())

    end_time_tmp = time.time()
    cc = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
    init.uniform_(cc,a=0,b=1)
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion_sup(outputs.sup, targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items,d= criterion_ssup(outputs, config, lambda_drloc)
                loss  =  (1-cc.item())*loss+loss_ssup*cc.item()
                loss += loss_ssup*cc.item()
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion_sup(outputs.sup, targets)
            if config.TRAIN.USE_DRLOC:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_drloc)
                loss = (1 - cc.item()) * loss + loss_ssup * cc.item()
                # loss += loss_ssup*cc.item()
                # print(cc.item())
                # loss = (loss-1.0).abs()+1.0
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            if config.TRAIN.USE_DRLOC:
                logger.info(f'weights: drloc {lambda_drloc:.4f}')
                logger.info(f' '.join(['%s: [%.4f]' % (key, value) for key, value in ssup_items.items()]))

        end_time_tmp = time.time()
    epoch_time = time.time() - start
    print(cc.item())

    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return OrderedDict([('loss', loss_meter.avg)])


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output.sup, target)
        acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    os.environ['Rank'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    main()
