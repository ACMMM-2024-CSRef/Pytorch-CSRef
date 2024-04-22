# Contrastive Semantic Alignment for Speech Referring Expression Comprehension

This repository contains the code for "Contrastive Semantic Alignment for Speech Referring Expression Comprehension (CSRef)".


## Data Preparation
1. Download speech referring expressions, speech encoder weights from Contrastive Semantic Alignment (CSA) stage, and pre-processing annotations JSON file to the data folder, following the path in [Google Drive](https://drive.google.com/drive/folders/1Z5dUdvLLCGbmbWaLLbHK2v7vJt1Rskxh?usp=sharing)
2. Download and unzip the [LibriSpeech](https://www.openslr.org/12) ASR dataset for CSA pre training to the `data/audios/` folder
3. Download and unzip the [train2014](http://images.cocodataset.org/zips/train2014.zip) images from [COCO](https://cocodataset.org/#home) to the `data/images` folder
4. Download [bert-base-uncased](https://huggingface.co/bert-base-uncased) and [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) from HuggingFace to the `data/weights/` folder


## Installation
- Clone this repo
- Create a conda virtual environment and activate it
```bash
conda create -n csref python=3.7.16
```
- Install Pytorch
- Install other packages in `requirements.txt`


## Training

### train for CSA stage
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 PORT=23450 bash tools/train_CSA.sh configs/csref_CSA_librispeech.py 4
```

### train for SREC stage
```bash
CUDA_VISIBLE_DEVICES=0 PORT=23450 bash tools/train_speech.sh configs/csref_refcoco+_speech.py 1
```

## Acknowledgement
Thanks to the following repos for their great works:
- [SimREC](https://github.com/luogen1996/SimREC)