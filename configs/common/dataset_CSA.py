from csref.config import LazyCall
from csref.datasets.dataset_CSA import LibriSpeechDataset


dataset = LazyCall(LibriSpeechDataset)(
    root_dir="data/audios/LibriSpeech",
    train_split="train",
    max_durations=None,
    use_trim=True,
    target_sample_rate=16000
)
