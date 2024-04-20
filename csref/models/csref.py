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

import numpy as np

import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False


class CSRef(nn.Module):
    def __init__(
            self,
            visual_backbone: nn.Module,
            speech_encoder: nn.Module,
            multi_scale_manner: nn.Module,
            fusion_manner: nn.Module,
            attention_manner: nn.Module,
            head: nn.Module,
    ):
        super(CSRef, self).__init__()
        self.visual_encoder = visual_backbone
        self.speech_encoder = speech_encoder
        self.multi_scale_manner = multi_scale_manner
        self.fusion_manner = fusion_manner
        self.attention_manner = attention_manner
        self.head = head

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    # def forward(self, x, y, audio_mask, det_label=None, seg_label=None):
    def forward(self, x, y, audio_mask, det_label=None):

        # vision and language encoding
        x = self.visual_encoder(x)
        y = self.speech_encoder(y, audio_mask)

        # vision and language fusion
        for i in range(len(self.fusion_manner)):
            x[i] = self.fusion_manner[i](x[i], y['flat_feat'])

        # multi-scale vision features
        x = self.multi_scale_manner(x)

        # multi-scale fusion layer
        top_feats, _, _ = self.attention_manner(y['flat_feat'], x[-1])

        bot_feats = x[0]

        # output
        if self.training:
            # loss, loss_det, loss_seg = self.head(top_feats, bot_feats, det_label, seg_label)
            # loss, loss_det, loss_seg = self.head(top_feats, bot_feats, det_label)
            # return loss, loss_det, loss_seg
            loss = self.head(top_feats, labels=det_label)
            return loss
        else:
            # box, mask = self.head(top_feats, bot_feats)
            # return box, mask
            box = self.head(top_feats)
            return box
