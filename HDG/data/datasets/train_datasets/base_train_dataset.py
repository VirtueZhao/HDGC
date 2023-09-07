import os
import gdown
import zipfile
import os.path as osp


class Datum:
    def __init__(self, img_path, class_name, class_label, domain_label):
        self._img_path = img_path
        self._class_name = class_name
        self._class_label = class_label
        self._domain_label = domain_label

    @property
    def img_path(self):
        return self._img_path

    @property
    def class_name(self):
        return self._class_name

    @property
    def class_label(self):
        return self._class_label

    @property
    def domain_label(self):
        return self._domain_label


class BaseTrainDataset:

    def __init__(self, dataset_dir, domains, data_url=None, train_data=None, class_name_to_class_label_mapping=None):
        self._dataset_dir = dataset_dir
        self._domains = domains
        self._data_url = data_url
        self._train_data = train_data
        self._num_classes = len(class_name_to_class_label_mapping.keys())
        self._class_names = class_name_to_class_label_mapping.keys()
        self._class_name_to_class_label_mapping = class_name_to_class_label_mapping
        self._class_label_to_class_name_mapping = self.get_class_label_to_class_name_mapping()

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def domains(self):
        return self._domains

    @property
    def data_url(self):
        return self._data_url

    @property
    def train_data(self):
        return self._train_data

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_name_to_class_label_mapping(self):
        return self._class_name_to_class_label_mapping

    @property
    def class_label_to_class_name_mapping(self):
        return self._class_label_to_class_name_mapping

    @property
    def num_classes(self):
        return self._num_classes

    def get_class_label_to_class_name_mapping(self):
        class_label_to_class_name_mapping = {self._class_name_to_class_label_mapping[class_name]: class_name
                                             for class_name in self._class_name_to_class_label_mapping.keys()}

        return class_label_to_class_name_mapping

    def download_data_from_gdrive(self, dst):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        gdown.download(self.data_url, dst, quiet=False)

        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(osp.dirname(dst))
        zip_ref.close()
        print("File Extracted to {}".format(osp.dirname(dst)))
