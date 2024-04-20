import json
import os
import random

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as Data

from csref.utils.distributed import is_main_process
from .utils import label2yolobox


class SpeechRefCOCODataSet(Data.Dataset):
    def __init__(self,
                 ann_path,
                 image_path,
                 audio_root,
                 # mask_path,
                 input_shape,
                 speakers,
                 flip_lr,
                 transforms,
                 candidate_transforms,
                 max_durations=None,
                 split="train",
                 dataset="refcoco",
                 use_trim=True,
                 target_sample_rate=16000,
                 # only_people=False
                 ):
        super(SpeechRefCOCODataSet, self).__init__()
        self.split = split

        # assert dataset in ['refcoco', 'refcoco+', 'refcocog', 'referit', 'vg', 'merge']
        assert dataset in ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech', 'srefface', 'srefface+', 'sreffaceg']
        self.dataset = dataset

        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.speakers = speakers
        self.use_trim = use_trim
        self.max_durations = max_durations

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list = json.load(open(ann_path[dataset], 'r'))
        total_refs_list = []
        # if dataset in ['vg', 'merge']:
        #     total_refs_list = json.load(open(ann_path['merge'], 'r')) + \
        #                       json.load(open(ann_path['refcoco+'], 'r')) + \
        #                       json.load(open(ann_path['refcocog'], 'r')) + \
        #                       json.load(open(ann_path['refcoco'], 'r'))


        splits = split.split('+')

        speech_refs_anno = []
        self.speech_refs_anno = []

        for split_ in splits:
            speech_refs_anno += stat_refs_list[split_]

        # if only_people:
        #     people_catids = [1] if dataset in ["refcoco", "refcoco+", "refcocog"] else [12, 13, 51, 52, 53, 66, 77, 88,
        #                                                                                 120, 122, 126, 135, 160, 187,
        #                                                                                 273]
        #     for ann in speech_refs_anno:
        #         cat_id = ann["cat_id"]
        #         if cat_id in people_catids:
        #             self.speech_refs_anno.append(ann)
        # else:
        #     self.speech_refs_anno = speech_refs_anno
        self.speech_refs_anno = speech_refs_anno

        refs = []

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)

        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)

        self.image_path = image_path[dataset]
        # self.mask_path = mask_path[dataset]
        self.input_shape = input_shape

        self.flip_lr = flip_lr if split == 'train' else False

        # Define run data size
        self.data_size = len(self.speech_refs_anno)

        if is_main_process():
            print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        if is_main_process():
            print('Finished!')
            print('')

        if split == 'train':
            self.candidate_transforms = candidate_transforms
        else:
            self.candidate_transforms = {}

        self.transforms = transforms


    def get_audio_by_sent_id(self, sent_id):
        # Randomly select a speaker
        speaker = self.speakers[np.random.choice(len(self.speakers))]
        path = os.path.join(self.audio_root, speaker, f"{sent_id}.wav")
        # Read audio and original sample rate
        wav, origin_sample_rate = sf.read(path, dtype="float32")
        # Resample to 16000Hz
        resampled_wav = librosa.resample(y=wav, orig_sr=origin_sample_rate, target_sr=self.target_sample_rate)
        # Remove the silent part of the head and tail of the speech
        if self.use_trim:
            resampled_wav, _ = librosa.effects.trim(resampled_wav)
        return resampled_wav

    def load_audio(self, idx):
        sent_ids = self.speech_refs_anno[idx]["sent_ids"]
        sent_id = sent_ids[np.random.choice(len(sent_ids))]

        audio = self.get_audio_by_sent_id(sent_id)

        if self.max_durations is not None:
            n_kept_frames = self.max_durations * self.target_sample_rate
            if len(audio) > n_kept_frames:
                audio = audio[0: n_kept_frames]

        return audio

    # def preprocess_info(self, img, mask, box, iid, aid, lr_flip=False):
    def preprocess_info(self, img, box, iid, aid, lr_flip=False):
        h, w, _ = img.shape
        imgsize = self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)

        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy, iid, aid)

        # sized_mask = None
        # if mask:
        #     mask = np.expand_dims(mask, -1).astype(np.float32)
        #     mask = cv2.resize(mask, (nw, nh))
        #     mask = np.expand_dims(mask, -1).astype(np.float32)
        #     sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        #     sized_mask[dy:dy + nh, dx:dx + nw, :] = mask
        #     sized_mask = np.transpose(sized_mask, (2, 0, 1))

        sized_box = label2yolobox(box, info_img, self.input_shape[0], lrflip=lr_flip)
        # return sized, sized_mask, sized_box, info_img
        return sized, sized_box, info_img

    def load_img_feats(self, idx):

        img_path = os.path.join(self.image_path, 'COCO_train2014_%012d.jpg' % self.speech_refs_anno[idx]['iid'])
        image = cv2.imread(img_path)


        # mask = None

        # box = np.array([self.speech_refs_anno[idx]['bbox']])

        # ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech', 'srefface', 'srefface+', 'sreffaceg']
        if self.dataset in ['refcoco_speech', 'refcoco+_speech', 'refcocog_speech']:
            box = np.array([self.speech_refs_anno[idx]['bbox']])
        elif self.dataset in ['srefface', 'srefface+', 'sreffaceg']:
            box = np.array([self.speech_refs_anno[idx]['fbox']])

        # box = np.array([self.speech_refs_anno[idx]['bbox']])

        # return image, mask, box, self.speech_refs_anno[idx]['mask_id'], self.speech_refs_anno[idx]['iid']
        # return image, box, self.speech_refs_anno[idx]['mask_id'], self.speech_refs_anno[idx]['iid']
        return image, box, self.speech_refs_anno[idx]['iid']

    def __getitem__(self, idx):

        audio_iter = self.load_audio(idx)

        sent_ids = self.speech_refs_anno[idx]["sent_ids"]
        sent_id = sent_ids[np.random.choice(len(sent_ids))]

        # image_iter, mask_iter, gt_box_iter, mask_id, iid = self.load_img_feats(idx)
        # image_iter, gt_box_iter, mask_id, iid = self.load_img_feats(idx)
        image_iter, gt_box_iter, iid = self.load_img_feats(idx)

        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        ops = None

        if len(list(self.candidate_transforms.keys())) > 0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]

        if ops is not None and ops != 'RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']

        flip_box = False
        if self.flip_lr and random.random() < 0.5:
            image_iter = image_iter[::-1]
            flip_box = True
        # image_iter, mask_iter, box_iter, info_iter = self.preprocess_info(image_iter, mask_iter, gt_box_iter.copy(),
        #                                                                   iid, sent_id, flip_box)
        # image_iter, mask_iter, box_iter, info_iter = self.preprocess_info(image_iter, gt_box_iter.copy(),
                                                                          # iid, sent_id, flip_box)
        image_iter, box_iter, info_iter = self.preprocess_info(image_iter, gt_box_iter.copy(),
                                                                          iid, sent_id, flip_box)

        return \
            audio_iter, \
            self.transforms(image_iter), \
            torch.from_numpy(box_iter).float(), \
            torch.from_numpy(gt_box_iter).float(), \
            np.array(info_iter)

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)
