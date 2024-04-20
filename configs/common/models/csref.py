import torch.nn as nn

from csref.config import LazyCall
from csref.models.csref import CSRef
from csref.models.backbones import CspDarkNet
from csref.models.heads import SREChead
from csref.models.speech_encoders.wav2vec2 import Wav2vec2
from csref.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention

model = LazyCall(CSRef)(
    visual_backbone=LazyCall(CspDarkNet)(
        pretrained=False,
        pretrained_weight_path="./data/weights/cspdarknet_coco.pth",
        freeze_backbone=True,
        multi_scale_outputs=True,
    ),
    speech_encoder=LazyCall(Wav2vec2)(
        hidden_size=768,
        flat_glimpses=1,
        dropout_rate=0.1,
        target_sr=16000,
        pretrained_path="data/weights/wav2vec2-base",
        freeze_model=True,
        use_one_hidden_state_as_feat=False,  # -25 for conv layer output, -24~-1 for Transformers layer output
        # when use_one_hidden_state_as_feat is True, specify the specific layer;
        # when False, specify the following several layers including this layer
        hidden_state_index=-20,
        use_att_flat_mask=True,
        fusion_times=1,
        # [:freeze_layers]
        freeze_layers=-1,
        short_cut=True,
    ),
    multi_scale_manner=LazyCall(MultiScaleFusion)(
        v_planes=(512, 512, 512),
        scaled=True
    ),
    fusion_manner=LazyCall(nn.ModuleList)(
        modules=[
            LazyCall(SimpleFusion)(v_planes=256, out_planes=512, q_planes=768),
            LazyCall(SimpleFusion)(v_planes=512, out_planes=512, q_planes=768),
            LazyCall(SimpleFusion)(v_planes=1024, out_planes=512, q_planes=768),
        ]
    ),
    attention_manner=LazyCall(GaranAttention)(
        d_q=768,
        d_v=512
    ),
    head=LazyCall(SREChead)(
        label_smooth=0.,
        num_classes=0,
        width=1.0,
        strides=[32, ],
        in_channels=[512, ],
        act="silu",
        depthwise=False
    )
)
