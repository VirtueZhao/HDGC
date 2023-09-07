# from .datasets.train_datasets.build_train_dataset import build_train_dataset
# from .datasets.test_datasets.build_test_dataset import build_test_dataset
from .datasets.domainnet.build_dataset import build_dataset
from .transforms import build_transform
import copy
import torch
import random
import numpy as np
from PIL import Image
from tabulate import tabulate
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


class TripletBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, batch_identity_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batch_identity_size = batch_identity_size
        self.num_pids_per_batch = self.batch_size // self.batch_identity_size

        self.index_dict = defaultdict(list)
        for index, datum in enumerate(data_source):
            self.index_dict[datum.class_name].append(index)
        self.pids = list(self.index_dict.keys())

        self.length = 0
        for pid in self.pids:
            indexs = self.index_dict[pid]
            num = len(indexs)
            if num < self.batch_identity_size:
                num = self.batch_identity_size
            self.length += num - num % self.batch_identity_size

    def __iter__(self):
        batch_indexs_dict = defaultdict(list)

        for pid in self.pids:
            indexs = copy.deepcopy(self.index_dict[pid])
            if len(indexs) < self.batch_identity_size:
                repeated_indexes = np.random.choice(indexs, size=self.batch_identity_size-len(indexs))
                indexs.extend(repeated_indexes)
            random.shuffle(indexs)

            batch_indexs = []
            for index in indexs:
                batch_indexs.append(index)
                if len(batch_indexs) == self.batch_identity_size:
                    batch_indexs_dict[pid].append(batch_indexs)
                    batch_indexs = []

        available_pids = copy.deepcopy(self.pids)
        final_indexs = []

        while len(available_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(available_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_indexs = batch_indexs_dict[pid].pop(0)
                final_indexs.extend(batch_indexs)
                if len(batch_indexs_dict[pid]) == 0:
                    available_pids.remove(pid)

        self.length = len(final_indexs)
        return iter(final_indexs)

    def __len__(self):
        return self.length


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    batch_identity_size=4,
    transform=None,
    is_train=True,
    dataset_wrapper=None
):
    if sampler_type == "TripletBatchSampler":
        sampler = TripletBatchSampler(data_source, batch_size, batch_identity_size)
    elif sampler_type == "RandomSampler":
        sampler = RandomSampler(data_source)
    elif sampler_type == "SequentialSampler":
        sampler = SequentialSampler(data_source)
    else:
        raise ValueError("Unknown Sampler Type :{}".format(sampler_type))

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_wrapper(data_source, transform),
        batch_size=256,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=dataset_wrapper(data_source, transform),
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #     drop_last=is_train,
    #     pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    # )

    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(self, cfg, custom_transform_train=None, custom_transform_test=None, dataset_wrapper=None):
        print("Build Data Manager")

        self.dataset = build_dataset(cfg)

        # Build Transform
        print("Build Transform")
        if custom_transform_train is None:
            transform_train = build_transform(cfg, is_train=True)
        else:
            transform_train = custom_transform_train

        if custom_transform_test is None:
            transform_test = build_transform(cfg, is_train=False)
        else:
            transform_test = custom_transform_test

        # Build Train Data Loader
        print("Build Train Data Loader")
        self.train_data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=self.dataset.train_data,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            batch_identity_size=cfg.DATALOADER.TRAIN.BATCH_IDENTITY_SIZE,
            transform=transform_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build Test Data Loader
        print("Build Test  Data Loader")
        self.test_data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=self.dataset.test_data,
            transform=transform_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        self._num_classes_train = self.dataset.train_data.num_classes
        self._num_source_domains = len(self.dataset.train_data.domains)
        self._num_classes_query = self.dataset.test_data.num_classes_query
        self._num_classes_gallery = self.dataset.test_data.num_classes_gallery
        self._class_label_to_class_name_mapping_train = self.dataset.train_data.class_label_to_class_name_mapping
        self._class_name_to_class_label_mapping_train = self.dataset.train_data.class_name_to_class_label_mapping
        self.show_dataset_summary(cfg)

    @property
    def num_classes_train(self):
        return self._num_classes_train

    @property
    def num_classes_query(self):
        return self._num_classes_query

    @property
    def num_classes_gallery(self):
        return self._num_classes_gallery

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def class_label_to_class_name_mapping_train(self):
        return self._class_label_to_class_name_mapping_train

    @property
    def class_name_to_class_label_mapping_train(self):
        return self._class_name_to_class_label_mapping_train

    def show_dataset_summary(self, cfg):
        dataset_table = [
            ["Source Dataset", cfg.DATASET.SOURCE_DOMAINS],
            ["Target Dataset", cfg.DATASET.TARGET_DOMAIN]
        ]

        domain_names = self.train_dataset.domain_info.keys()
        for domain_name in domain_names:
            dataset_table.append([domain_name, f"{self.train_dataset.domain_info[domain_name]:,}"])

        dataset_table.extend([
            ["# Train Identity", f"{self.num_classes_train:,}"],
            ["# Train   Data", f"{len(self.train_dataset.train_data):,}"],
            ["# Query   Data", f"{len(self.test_dataset.query_data):,}"],
            ['# Query Identity', f"{self.num_classes_query:,}"],
            ["# Gallery Data", f"{len(self.test_dataset.gallery_data):,}"],
            ['# Gallery Identity', f"{self.num_classes_gallery:,}"]
        ])

        print(tabulate(dataset_table))


class DatasetWrapper(Dataset):

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        datum = self.data_source[index]

        output = {
            "img_path": datum.img_path,
            "class_name": datum.class_name,
            "class_label": datum.class_label,
            "domain_label": datum.domain_label,
            "img": self.transform(Image.open(datum.img_path).convert("RGB"))
        }

        return output
