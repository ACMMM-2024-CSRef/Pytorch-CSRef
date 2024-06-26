"""
    train
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PORT=23450 bash tools/train_speech.sh configs/csref_srefface+.py 1
"""

from csref.config import LazyCall
from .common.train import train
from .common.optim import optim
from .common.models.csref import model
from .common.dataset_speech import dataset
from transformers import Wav2Vec2FeatureExtractor

# dataset config
dataset.dataset = "srefface+"
dataset.ann_path["srefface+"] = "./data/anns/face_centric/srefface+.json"
dataset.image_path["srefface+"] = "./data/images/train2014"
dataset.audio_root = "./data/audios/refcoco+_keepcomma"
dataset.speakers = ['en-US-SteffanNeural']

dataset.max_durations = None
dataset.use_trim = True  # remove silence

# train config
train.batch_size = 8
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 8
train.sync_bn.enabled = False

train.audio_encoder_ckpt_path = "data/weights/CSA_speech_encoder.pth"
train.output_dir = "./output/csref_srefface+"  #

train.data.num_workers = 8
train.epochs = 40

# optimization config
optim.lr = train.base_lr

# model config
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path = "./data/weights/cspdarknet_coco.pth"
model.speech_encoder.short_cut = True
model.speech_encoder.pretrained_path = "data/weights/wav2vec2-base"
# model.speech_encoders.pretrained_path = "data/weights/wav2vec2-base-960h"

model.speech_encoder.freeze_model = True
model.speech_encoder.use_one_hidden_state_as_feat = False
model.speech_encoder.hidden_state_index = -13
model.speech_encoder.use_att_flat_mask = True
model.speech_encoder.fusion_times = 13

# -24 represents all layers are not frozen,
# -1 represents the last layer is not frozen,
# 24 represents all layers are frozen
model.speech_encoder.freeze_layers = -1

preprocessor = LazyCall(Wav2Vec2FeatureExtractor.from_pretrained)(
    pretrained_model_name_or_path=model.speech_encoder.pretrained_path
)
