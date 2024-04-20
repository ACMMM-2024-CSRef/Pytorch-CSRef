import albumentations as A
from torchvision.transforms import transforms

from csref.config import LazyCall
from csref.datasets.dataset_speech import SpeechRefCOCODataSet

from .train import train

dataset = LazyCall(SpeechRefCOCODataSet)(
    dataset="refcoco_speech",
    # dataset="refcoco+_speech",
    # dataset="refcocog_speech",
    # dataset="srefface",
    # dataset="srefface+",
    # dataset="sreffaceg",

    # only_people=False,

    max_durations=None,

    ann_path={
        'refcoco_speech': './data/anns/general_object/refcoco_sent_ids.json',
        'refcoco+_speech': './data/anns/general_object/refcoco+_sent_ids.json',
        'refcocog_speech': './data/anns/general_object/refcocog_sent_ids.json',
        'srefface': './data/anns/face_centric/refcoco_sent_ids.json',
        'srefface+': './data/anns/face_centric/refcoco+_sent_ids.json',
        'sreffaceg': './data/anns/face_centric/refcocog_sent_ids.json',
    },
    image_path={
        'refcoco_speech': './data/images/train2014',
        'refcoco+_speech': './data/images/train2014',
        'refcocog_speech': './data/images/train2014',
        'srefface': './data/images/train2014',
        'srefface+': './data/images/train2014',
        'sreffaceg': './data/images/train2014',
    },

    audio_root="./data/audios/refcoco",
    # audio_root="./data/audios/refcoco+_keepcomma",
    # audio_root="./data/audios/refcocog_keepcomma",

    speakers=['en-US-SteffanNeural'],

    # mask_path={
    #     'refcoco': './data/masks/refcoco',
    #     'refcoco+': './data/masks/refcoco+',
    #     'refcocog': './data/masks/refcocog',
    #     # 'referit': './data/masks/refclef'
    # },

    # original input image shape
    input_shape=[416, 416],
    flip_lr=False,

    # basic transforms
    transforms=LazyCall(transforms.Compose)(
        transforms=[
            LazyCall(transforms.ToTensor)(),
            LazyCall(transforms.Normalize)(
                mean=train.data.mean,
                std=train.data.std,
            )
        ]
    ),

    # candidate transforms
    candidate_transforms={
        # "RandAugment": RandAugment(2, 9),
        # "ElasticTransform": A.ElasticTransform(p=0.5),
        # "GridDistortion": A.GridDistortion(p=0.5),
        # "RandomErasing": transforms.RandomErasing(
        #     p = 0.3,
        #     scale = (0.02, 0.2),
        #     ratio=(0.05, 8),
        #     value="random",
        # )
    },

    # datasets splits
    split="train",

    use_trim=True,  # TODO 【实验】可以对比

    target_sample_rate=16000
)
