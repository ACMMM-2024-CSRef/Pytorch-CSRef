import time
import numpy as np

import torch
from csref.utils.metric import AverageMeter
from csref.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, writer, epoch, logger, rank, audio_preprocessor, text_tokenizer, save_ids=None,
             prefix='Val', ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    mask_aps = {}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item] = []
    meters = [batch_time, data_time, losses]
    meters_dict = {meter.name: meter for meter in meters}

    with torch.no_grad():
        end = time.time()
        for idx, (audio_iter, text_iter) in enumerate(data_loader):
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

            loss = model(audio_iter, audio_mask_iter, text_iter, text_mask_iter)

            losses.update(loss.item(), audio_iter.size(0))

            reduce_meters(meters_dict, rank, cfg)

            if (idx % cfg.train.log_period == 0 or idx == (len(data_loader) - 1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            batch_time.update(time.time() - end)
            end = time.time()

        if is_main_process() and writer is not None:
            writer.add_scalar("Loss", losses.avg_reduce, global_step=epoch)
        logger.info(f' * Loss {losses.avg_reduce:.4f}')

    if ema is not None:
        ema.restore()
    return losses.avg_reduce
