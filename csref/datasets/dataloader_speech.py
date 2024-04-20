import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    audios = [item[0] for item in batch]
    images = [item[1] for item in batch]
    boxs = [item[2] for item in batch]
    gt_boxs = [item[3] for item in batch]
    infos = [item[4] for item in batch]
    return [audios,
            torch.stack(images),
            torch.stack(boxs),
            torch.stack(gt_boxs),
            # torch.stack(infos),
            infos,
            ]


def build_train_speech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=True, drop_last=False):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    assert cfg.train.batch_size % num_tasks == 0
    assert dist.is_initialized()

    train_micro_batch_size = cfg.train.batch_size // num_tasks

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        shuffle=shuffle,
        rank=global_rank,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=train_micro_batch_size,
        sampler=train_sampler,
        num_workers=cfg.train.data.num_workers,
        pin_memory=cfg.train.data.pin_memory,
        drop_last=drop_last,
        collate_fn=my_collate_fn
    )
    return data_loader


def build_test_speech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=False, drop_last=False):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    assert cfg.train.evaluation.eval_batch_size % num_tasks == 0
    assert dist.is_initialized()

    eval_micro_batch_size = cfg.train.evaluation.eval_batch_size // num_tasks

    if cfg.train.evaluation.sequential:
        eval_micro_batch_size = cfg.train.evaluation.eval_batch_size
        eval_sampler = SequentialSampler(dataset)
    else:
        eval_sampler = DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            shuffle=shuffle,
            rank=global_rank,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=eval_micro_batch_size,
        sampler=eval_sampler,
        num_workers=cfg.train.data.num_workers,
        pin_memory=cfg.train.data.pin_memory,
        drop_last=drop_last,
        collate_fn=my_collate_fn
    )
    return data_loader
