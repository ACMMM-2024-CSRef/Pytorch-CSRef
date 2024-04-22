"""
dataset

LibriSpeech
    |--train-clean-100
    |--train-clean-360
    |--train-other-500
    |--dev-clean
    |--dev-other
    |--test-clean
    |--test-other
"""

import os

import librosa
from torch.utils.data import Dataset
from csref.utils.distributed import is_main_process

import soundfile as sf


class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, train_split='train', max_durations=None,
                 use_trim=True,
                 target_sample_rate=16000):
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.use_trim = use_trim
        self.max_durations = max_durations

        if train_split == 'train':
            self.splits = [
                'train-clean-100',
                'train-clean-360',
                'train-other-500'
            ]
        elif train_split == 'val':
            self.splits = [
                'dev-clean',
                'dev-other',
                'test-clean',
                'test-other'
            ]
        elif train_split == 'test':
            self.splits = [
                'test-clean',
                'test-other'
            ]
        else:
            raise ValueError(f"Invalid split: {train_split}")

        self.speech_files = []
        self.transcripts = []

        self._load_data()

        if is_main_process():
            print(f'====== Dataset {train_split} loaded! ======')
            print('Max durations:', max_durations, '\n',
                  'Trimmed:', use_trim, '\n',
                  'Target sample rate:', target_sample_rate, '\n',
                  'num of samples:', len(self.speech_files)
                  )
            print(f'====== Dataset {train_split} loaded! ======')

    def _load_data(self):
        for split in self.splits:
            split_path = os.path.join(self.root_dir, split)

            for speaker in os.listdir(split_path):
                speaker_path = os.path.join(split_path, speaker)
                for chapter in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter)

                    transcripts_dict = {}

                    for file in os.listdir(chapter_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(chapter_path, file)

                            with open(file_path, 'r') as f:
                                for line in f:
                                    key, value = line.strip().split(' ', 1)
                                    transcripts_dict[key] = value
                    for file in os.listdir(chapter_path):
                        if file.endswith('.flac'):
                            file_path = os.path.join(chapter_path, file)
                            self.speech_files.append(file_path)

                            # Get transcript
                            transcript = transcripts_dict[file.split('.')[0]]
                            self.transcripts.append(transcript)

    def load_audio(self, idx):
        speech_file = self.speech_files[idx]

        audio = self.get_audio_by_path(speech_file)

        if self.max_durations is not None:
            n_kept_frames = self.max_durations * self.target_sample_rate
            if len(audio) > n_kept_frames:
                audio = audio[0: n_kept_frames]

        return audio

    def get_audio_by_path(self, path):

        wav, origin_sample_rate = sf.read(path, dtype="float32")

        resampled_wav = librosa.resample(y=wav, orig_sr=origin_sample_rate, target_sr=self.target_sample_rate)

        if self.use_trim:
            resampled_wav, _ = librosa.effects.trim(resampled_wav)
        return resampled_wav

    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, idx):

        waveform = self.load_audio(idx)
        transcript = self.transcripts[idx]

        return waveform, transcript
