import os.path

import numpy as np
import os.path as osp

import torch

from .build_dataset import DATASET_REGISTRY
from .base_dataset import Datum, BaseDataset


@DATASET_REGISTRY.register()
class DomainNet(BaseDataset):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """
    def __init__(self, cfg):
        self._dataset_dir = "domainnet"
        self._domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)

        self.domain_info = {}
        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, [cfg.DATASET.TARGET_DOMAIN])

        # seen_list = osp.join(self._dataset_dir, "train_classes.npy")
        self.seen_list = list(np.load(osp.join(self._dataset_dir, "train_classes.npy"))) + \
                         list(np.load(osp.join(self._dataset_dir, "val_classes.npy")))
        self.unseen_list = list(np.load(osp.join(self._dataset_dir, "test_classes.npy")))

        self.full_classes = self.seen_list + self.unseen_list
        self.seen = torch.LongTensor([self.full_classes.index(k) for k in self.seen_list])
        self.unseen = torch.LongTensor([self.full_classes.index(k) for k in self.unseen_list])
        self.full_classes_idx = torch.cat([self.seen, self.unseen], dim=0)

        self.attributes_dict = np.load(os.path.join(self._dataset_dir, "w2v_domainnet.npy"), allow_pickle=True,
                                       encoding="latin1").item()

        self.attribute_size = len(self.attributes_dict[list(self.attributes_dict.keys())[0]])

        for key in self.attributes_dict.keys():
            self.attributes_dict[key] = torch.Tensor(self.attributes_dict[key])

        for index, key in enumerate(self.full_classes):
            self.attributes_dict[index] = self.attributes_dict[key]

        train_data = self.read_train_data(cfg.DATASET.SOURCE_DOMAINS)
        test_data = self.read_test_data(cfg.DATASET.TARGET_DOMAIN)

        super().__init__(dataset_dir=self._dataset_dir, domains=self._domains, train_data=train_data, test_data=test_data)

    def read_train_data(self, source_domains):

        def _load_img_paths(directory):
            img_paths = []
            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img_path, class_label = line.split(" ")
                    class_name = img_path.split('/')[1]
                    if class_name in self.seen_list:
                        img_path = osp.join(self._dataset_dir, img_path)
                        class_label = self.seen_list.index(class_name)
                        class_attribute = self.attributes_dict[class_name].unsqueeze(0)
                        img_paths.append((img_path, class_name, class_label, class_attribute))

            return img_paths

        img_datums = []

        for domain_label, domain_name in enumerate(source_domains):
            data_dir = osp.join(self._dataset_dir, domain_name + "_train.txt")
            img_path_class_label_list = _load_img_paths(data_dir)
            self.domain_info[domain_name] = len(img_path_class_label_list)

            for img_path, class_name, class_label, class_attribute in img_path_class_label_list:
                img_datum = Datum(
                    img_path=img_path,
                    class_name=class_name,
                    class_label=class_label,
                    class_attribute=class_attribute,
                    domain_label=domain_label
                )
                img_datums.append(img_datum)

        return img_datums

    def read_test_data(self, target_domain):

        def _load_img_paths(directory):
            img_paths = []
            with open(directory, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img_path, class_label = line.split(" ")
                    class_name = img_path.split('/')[1]
                    if class_name in self.unseen_list:
                        img_path = osp.join(self._dataset_dir, img_path)
                        class_label = self.unseen_list.index(class_name)
                        class_attribute = self.attributes_dict[class_name].unsqueeze(0)
                        img_paths.append((img_path, class_name, class_label, class_attribute))

            return img_paths

        domain_label = 0
        domain_name = target_domain

        data_dir = osp.join(self._dataset_dir, domain_name + "_train.txt")
        img_path_class_label_list = _load_img_paths(data_dir)
        data_dir = osp.join(self._dataset_dir, domain_name + "_test.txt")
        img_path_class_label_list += _load_img_paths(data_dir)

        img_datums = []

        for img_path, class_name, class_label, class_attribute in img_path_class_label_list:
            img_datum = Datum(
                img_path=img_path,
                class_name=class_name,
                class_label=class_label,
                class_attribute=class_attribute,
                domain_label=domain_label
            )
            img_datums.append(img_datum)

        return img_datums
