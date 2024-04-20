# Contrastive Semantic Alignment for Speech Referring Expression Comprehension

This repository contains the code for "Contrastive Semantic Alignment for Speech Referring Expression Comprehension (CSRef)".


## Data Preparation
Please refer to [DATA_PRE_README.md](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md) for data preparation, 
and download the speech data from [(to be release)]()




## Training

### train for SREC stage
```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PORT=23450 bash tools/train_speech.sh configs/csref_refcoco+.py 1
```

## Acknowledgement
Thanks to the following repos for their great works:
- [SimREC](https://github.com/luogen1996/SimREC)