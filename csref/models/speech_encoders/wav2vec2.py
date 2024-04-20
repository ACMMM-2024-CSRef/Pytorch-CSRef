import torch
import torch.nn as nn

from csref.layers.sa_layer import AttFlat

from transformers import Wav2Vec2Model
from transformers.activations import GELUActivation


class GELUConv(nn.Module):
    """A Conv2d -> Batchnorm -> GELU activation"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, shortcut=False, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = GELUActivation()

        if shortcut:
            assert in_channels == out_channels

        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return self.act(self.bn(x + self.conv(x) if self.add else self.conv(x)))


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class Wav2vec2(nn.Module):
    def __init__(
            self,
            hidden_size=1024,
            flat_glimpses=1,
            dropout_rate=0.1,
            target_sr=16000,
            pretrained_path="/data/weights/wav2vec2-base-960h",
            freeze_model=False,
            use_one_hidden_state_as_feat=True,
            hidden_state_index=-1,
            use_att_flat_mask=True,
            fusion_times=1,
            freeze_layers=None,  # freeze the first few layers of encoder.layers
            short_cut=False,
    ):
        super(Wav2vec2, self).__init__()

        self.hidden_size = hidden_size
        self.target_sample_rate = target_sr
        self.use_one_hidden_state_as_feat = use_one_hidden_state_as_feat

        # The index of hidden states to use as features.
        # When use_one_hidden_state_as_feat is True, it refers to outputs.hidden_states[hidden_state_index];
        # when False, it refers to outputs.hidden_states[hidden_state_index:]
        self.hidden_state_index = hidden_state_index
        self.use_att_flat_mask = use_att_flat_mask
        self.fusion_times = fusion_times

        self.model = Wav2Vec2Model.from_pretrained(pretrained_path, gradient_checkpointing=False)

        if freeze_model:
            if freeze_layers is not None:
                self.model.masked_spec_embed.required_grad = False
                self.frozen(self.model.feature_extractor)
                self.frozen(self.model.feature_projection)
                self.frozen(self.model.encoder.pos_conv_embed)
                self.frozen(self.model.encoder.layer_norm)
                self.frozen(self.model.encoder.layers[:freeze_layers])
            else:
                self.model.freeze_feature_encoder()

        if not use_one_hidden_state_as_feat:
            self.fusion_modules = GELUConv(
                in_channels=abs(hidden_state_index),
                out_channels=fusion_times,
                ksize=1,
                stride=1,
                shortcut=short_cut
            )
        self.att_flat = AttFlat(hidden_size, flat_glimpses, dropout_rate)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, audio, mask):

        output = self.model(
            input_values=audio,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        feat = None
        if self.use_one_hidden_state_as_feat:
            hidden_state = output.hidden_states[self.hidden_state_index]  # large[b, len ,c(1024)]
            feat = hidden_state  # (batch, len, channel)
        else:
            feat = torch.stack(output.hidden_states[self.hidden_state_index:], 1)  # (batch, n_hidden, len, channel)
            feat = self.fusion_modules(feat)  # (batch, 1, len, channel)
            feat = torch.flatten(feat, 1, 2)  # (batch, len, channel)

        mask_flip_bool = None
        if self.use_att_flat_mask:
            first_attention_0 = self.model._get_feature_vector_attention_mask(output.hidden_states[-1].shape[1],
                                                                              mask)  # [b, len]
            mask_flip_bool = make_mask(first_attention_0.unsqueeze(2))  # (batch, 1, 1, len)
            mask_flip_bool = torch.cat([mask_flip_bool for i in range(self.fusion_times)], 3)

        flat_feat = self.att_flat(feat, mask_flip_bool)

        return {
            'flat_feat': flat_feat,
            'feat': feat,
        }
