import sys
import glob
import os.path as osp
from collections import OrderedDict
from .base_train_dataset import Datum, BaseTrainDataset
from .build_train_dataset import TRAIN_DATASET_REGISTRY


@TRAIN_DATASET_REGISTRY.register()
class Market1501(BaseTrainDataset):

    def __init__(self, cfg):
        self._dataset_dir = "Market1501"
        self._domains = ["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6"]
        self._data_url = "https://drive.google.com/uc?id=1f__4ttSaNXLH9XQkRoBKX9SPTALmK2VT"
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self.domain_info = {}

        if not osp.exists(self._dataset_dir):
            dst = osp.join(dataset_path, "Market1501.zip")
            self.download_data_from_gdrive(dst)

        train_data, class_name_to_class_label_mapping = self.read_train_data(self._domains)
        super().__init__(dataset_dir=self._dataset_dir, domains=self._domains, data_url=self._data_url,
                         train_data=train_data, class_name_to_class_label_mapping=class_name_to_class_label_mapping)

    def read_train_data(self, input_domains):
        datums = []

        class_name_to_class_label_mapping = OrderedDict()
        class_label = 0

        for domain_label, domain_name in enumerate(input_domains):
            domain_dir = osp.join(self._dataset_dir, domain_name)
            images = glob.glob(osp.join(domain_dir, "*.jpg"))
            self.domain_info[domain_name] = len(images)

            for img_path in images:
                if sys.platform == "linux":
                    class_name = img_path.split("/")[-1].split("_")[0]
                else:
                    class_name = img_path.split("\\")[-1].split("_")[0]

                if class_name in class_name_to_class_label_mapping:
                    current_class_label = class_name_to_class_label_mapping[class_name]
                else:
                    class_name_to_class_label_mapping[class_name] = class_label
                    current_class_label = class_label
                    class_label += 1

                datum = Datum(
                    img_path=img_path,
                    class_name=class_name,
                    class_label=current_class_label,
                    domain_label=domain_label
                )
                datums.append(datum)

        return datums, class_name_to_class_label_mapping
