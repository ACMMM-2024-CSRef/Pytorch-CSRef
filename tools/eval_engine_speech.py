import os
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from csref.config import instantiate, LazyConfig
from csref.datasets.dataloader_speech import build_test_speech_loader
from csref.datasets.utils import yolobox2label
from csref.models.utils import batch_box_iou
from csref.utils.env import seed_everything
from csref.utils.logger import create_logger
from csref.utils.metric import AverageMeter
from csref.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, writer, epoch, logger, rank, audio_preprocessor, save_ids=None, prefix='Val',
             ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')

    sample_time = AverageMeter('Sample', ':6.5f')

    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps = {}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item] = []
    meters = [batch_time, data_time, losses, box_ap, mask_ap, inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}

    with torch.no_grad():
        end = time.time()
        for idx, (audio_iter, image_iter, box_iter, gt_box_iter, info_iter) in enumerate(data_loader):
            batch_audio = audio_preprocessor(audio_iter, padding=True, max_length=None, truncation=False,
                                             pad_to_multiple_of=None, return_attention_mask=True, return_tensors="pt",
                                             sampling_rate=cfg.dataset.target_sample_rate)
            audio_iter = batch_audio.input_values
            audio_mask_iter = batch_audio.attention_mask

            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)
            box = model(image_iter, audio_iter, audio_mask_iter)

            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter = gt_box_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()

            info_iter = np.array(info_iter)

            # predictions to ground-truth
            for i in range(len(gt_box_iter)):
                box[i] = yolobox2label(box[i], info_iter[i])

            box_iou = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box)).cpu().numpy()

            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])

            reduce_meters(meters_dict, rank, cfg)

            if (idx % cfg.train.log_period == 0 or idx == (len(data_loader) - 1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    # f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'BoxIoU@0.5 {box_ap.val:.4f} ({box_ap.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            sample_time.update((time.time() - end) / cfg.train.evaluation.eval_batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

        if is_main_process() and writer is not None:
            writer.add_scalar("Acc/BoxIoU_0.5", box_ap.avg_reduce, global_step=epoch)

        logger.info(f' * BoxIoU@0.5 {box_ap.avg_reduce:.3f}  Sample_time {sample_time.avg:.5f}')

    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, mask_ap.avg_reduce


def main(cfg):
    global best_det_acc, best_seg_acc
    best_det_acc, best_seg_acc = 0., 0.

    audio_preprocessor = instantiate(cfg.preprocessor)

    # build single or multi-datasets for validation
    loaders = []
    prefixs = ['val_eval']
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_speech_loader(cfg, val_set, shuffle=False, drop_last=False)
    loaders.append(val_loader)

    if cfg.dataset.dataset in ['refcoco_speech', 'refcoco+_speech']:
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_test_speech_loader(cfg, testA_dataset, shuffle=False, drop_last=False)
        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        testB_loader = build_test_speech_loader(cfg, testB_dataset, shuffle=False, drop_last=False)
        prefixs.extend(['testA', 'testB'])
        loaders.extend([testA_loader, testB_loader])
    elif cfg.dataset.dataset in ['srefface', 'srefface+']:
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_test_speech_loader(cfg, testA_dataset, shuffle=False, drop_last=False)
        prefixs.extend(['testA'])
        loaders.extend([testA_loader])
    else:
        cfg.dataset.split = "test"
        test_dataset = instantiate(cfg.dataset)
        test_loader = build_test_speech_loader(cfg, test_dataset, shuffle=False, drop_last=False)
        prefixs.append('test')
        loaders.append(test_loader)

    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    torch.cuda.set_device(dist.get_rank())
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda())
    model_without_ddp.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if cfg.train.amp:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    # if is_main_process():
    #     writer = SummaryWriter(log_dir=cfg.train.output_dir)
    # else:
    #     writer = None
    writer = None

    save_ids = np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None
    for data_loader, prefix in zip(loaders, prefixs):
        box_ap, mask_ap = validate(
            cfg=cfg,
            model=model,
            data_loader=data_loader,
            writer=writer,
            epoch=0,
            logger=logger,
            rank=dist.get_rank(),
            audio_preprocessor=audio_preprocessor,
            save_ids=save_ids,
            prefix=prefix)
        logger.info(f' * BoxIoU@0.5 {box_ap:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="csref_SREC")
    parser.add_argument('--config', type=str, required=True, default='./configs/csref_refcoco+_speech.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
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
    os.makedirs(os.path.join(output_dir, "eval_result_log"), exist_ok=True)
    checkpoint_name = os.path.basename(args.eval_weights)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank(), name=f"eval_{checkpoint_name}")

    # Refine cfg for evaluation
    cfg.train.resume_path = args.eval_weights
    logger.info(f"Running evaluation from specific checkpoint {cfg.train.resume_path}......")

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "eval_result_log", "config_eval.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
