import os
import sys
import time
import datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from csref.config import LazyConfig, instantiate

from csref.datasets.dataloader_CSA import build_train_csa_loader, build_test_csa_loader

from csref.scheduler.build import build_lr_scheduler
from csref.utils.model_ema import EMA
from csref.utils.logger import create_logger
from csref.utils.env import seed_everything
from csref.utils.metric import AverageMeter
from csref.utils.distributed import reduce_meters, is_main_process, cleanup_distributed
from csref.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume_helper

from tools.eval_CSA import validate

from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer


def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, scalar, writer, epoch, rank, audio_preprocessor,
                    text_tokenizer, ema=None):
    model.train()
    data_loader.sampler.set_epoch(epoch)

    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    meters = [batch_time, data_time, losses]
    meters_dict = {meter.name: meter for meter in meters}

    start = time.time()
    end = time.time()
    for idx, (audio_iter, text_iter) in enumerate(data_loader):
        data_time.update(time.time() - end)

        # ref_iter = ref_iter.cuda(non_blocking=True)
        batch_audio = audio_preprocessor(raw_speech=audio_iter, padding=True, max_length=None, truncation=False,
                                         pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                         sampling_rate=cfg.dataset.target_sample_rate)
        batch_text = text_tokenizer.batch_encode_plus(
            text_iter,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors='pt',
            return_attention_mask=True
        )
        audio_iter = batch_audio.input_values
        audio_mask_iter = batch_audio.attention_mask

        text_iter = batch_text.input_ids
        text_mask_iter = batch_text.attention_mask

        if scalar is not None:
            with torch.cuda.amp.autocast():
                loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)
        else:
            loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)

        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            scalar.update()
        else:
            loss.backward()
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update_params()

        losses.update(loss.item(), audio_iter.size(0))

        reduce_meters(meters_dict, rank, cfg)
        if is_main_process():
            global_step = epoch * num_iters + idx
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr/train", lr, global_step=global_step)

        if idx % cfg.train.log_period == 0 or idx == len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_iters}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Mem {memory_used:.0f}MB')

        batch_time.update(time.time() - end)
        end = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(cfg):
    global lowest_loss
    lowest_loss = sys.float_info.max


    audio_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model.speech_encoder.pretrained_path)
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.pretrained_path)

    # build training dataset and dataloader
    cfg.dataset.train_split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader = build_train_csa_loader(
        cfg,
        train_set,
        shuffle=True,
        drop_last=True
    )

    # build validation dataset and dataloader
    cfg.dataset.train_split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_csa_loader(
        cfg,
        val_set,
        shuffle=False,
        drop_last=False,
    )

    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # model ema
    ema = None

    torch.cuda.set_device(dist.get_rank())
    if cfg.train.sync_bn.enabled:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted model to use Synchronized BatchNorm.")
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    start_epoch = 0

    if cfg.train.auto_resume.enabled:
        resume_file = auto_resume_helper(cfg.train.output_dir)
        if resume_file:
            if cfg.train.resume_path:
                logger.warning(f"auto-resume changing resume file from {cfg.train.resume_path} to {resume_file}")
            cfg.train.resume_path = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.train.output_dir}, ignoring auto resume')

    if cfg.train.resume_path:
        start_epoch = load_checkpoint(cfg, model_without_ddp, optimizer, scheduler, logger)

    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda())
        logger.warning("loading pretrained weight for finetuning, ignoring resume training, reset start epoch to 0")
        msg = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        start_epoch = 0
        logger.info("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                    "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if cfg.train.amp.enabled:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None


    for epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        train_one_epoch(cfg, model, optimizer, scheduler, train_loader, scalar, writer, epoch, dist.get_rank(),
                        audio_preprocessor, text_tokenizer, ema)
        loss = validate(cfg, model, val_loader, writer, epoch, logger, dist.get_rank(),
                        save_ids=None, audio_preprocessor=audio_preprocessor, text_tokenizer=text_tokenizer,
                        ema=ema)

        # save checkpoints
        if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
            logger.info(f"saving checkpoints......")
            if is_main_process():
                if ema is not None:
                    ema.apply_shadow()
                # periodically save checkpoint
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
                if ema is not None:
                    ema.restore()
            logger.info(f"checkpoints saved !!!")

        # # save best checkpoint
        # fix bug
        if is_main_process():
            if ema is not None:
                ema.apply_shadow()
            if loss < lowest_loss:
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
                lowest_loss = loss
                logger.info(f"best_det_checkpoints saved !!!")
            if ema is not None:
                ema.restore()

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="csref_CSA")
    parser.add_argument('--config', type=str, required=True, default='./configs/csref_CSA_librispeech.py')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Environments setting
    seed_everything(cfg.train.seed)

    # Distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend,
        init_method=cfg.train.ddp.init_method,
        world_size=world_size,
        rank=rank
    )
    torch.distributed.barrier()

    # Path setting
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
