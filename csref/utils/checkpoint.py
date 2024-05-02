# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from csref.utils.distributed import is_main_process


def load_checkpoint(cfg, model, optimizer, scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.train.resume_path}....................")
    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda())
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]
    logger.info("==> loaded checkpoint from {}\n".format(cfg.train.resume_path) +
                "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))
    return start_epoch + 1


def load_checkpoint_for_audio_encoder(cfg, model, logger):
    logger.info(f"==============> Reloading Audio Encoder ckpt form {cfg.train.audio_encoder_ckpt_path}....................")
    checkpoint = torch.load(cfg.train.audio_encoder_ckpt_path, map_location=lambda storage, loc: storage.cuda())
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("speech_encoder."):
            new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(msg)


def save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, det_best=False, seg_best=False):
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'lr': optimizer.param_groups[0]["lr"]
    }
    save_path = os.path.join(cfg.train.output_dir, f'ckpt_epoch_{epoch}.pth')

    if not os.path.exists(save_path):
        torch.save(save_state, save_path)

    # save last checkpoint
    last_checkpoint_path = os.path.join(cfg.train.output_dir, f'last_checkpoint.pth')
    torch.save(save_state, last_checkpoint_path)  

    # save the best detection model
    if det_best:
        det_best_model_path = os.path.join(cfg.train.output_dir, f'det_best_model.pth')
        torch.save(save_state, det_best_model_path)

    # save the best segmentation model
    if seg_best:
        seg_best_model_path = os.path.join(cfg.train.output_dir, f'seg_best_model.pth')
        torch.save(save_state, seg_best_model_path)


def save_for_predict(cfg, dataset):
    save_state = {
        'max_token': dataset.max_token,
        'token_to_ix': dataset.token_to_ix,
        'input_shape': dataset.input_shape
    }
    save_path = os.path.join("inference_parameters", f'parameters_for_{cfg.dataset.dataset}_predict.pth')
    torch.save(save_state, save_path)


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        resume_file = os.path.join(output_dir, "last_checkpoint.pth")
    else:
        resume_file = None

    return resume_file
