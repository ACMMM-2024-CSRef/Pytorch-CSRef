"""
    train
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,2,3,4 PORT=23450 bash tools/train_CSA.sh configs/csref_CSA_librispeech.py 4
"""
from .common.dataset_CSA import dataset
from .common.train import train
from .common.optim import optim
from .common.models.csref_CSA import model

# Refine data path depend your own need
dataset.root_dir = "data/audios/LibriSpeech"

# Refine training cfg
train.output_dir = "./output/csref_CSA_librispeech"
train.batch_size = 32
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 32
train.sync_bn.enabled = False
train.epochs = 9

train.amp.enable=True

# train.min_lr = 5.9e-6  # convergence when batch size is 32

train.ema.enabled = False

# optim config
optim.lr = train.base_lr

# model config
model.speech_encoder.pretrained_path = "data/weights/wav2vec2-base"
model.text_encoder.pretrained_path = "data/weights/bert-base-uncased"


