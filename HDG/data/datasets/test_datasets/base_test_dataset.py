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


class BaseTestDataset:

    def __init__(self, dataset_dir, data_url, query_data=None, gallery_data=None):
        self._dataset_dir = dataset_dir
        self._data_url = data_url
        self._query_data = query_data
        self._gallery_data = gallery_data
        self._num_classes_query = self.get_num_classes(self._query_data)
        self._num_classes_gallery = self.get_num_classes(self._gallery_data)

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def data_url(self):
        return self._data_url

    @property
    def query_data(self):
        return self._query_data

    @property
    def gallery_data(self):
        return self._gallery_data

    @property
    def num_classes_query(self):
        return self._num_classes_query

    @property
    def num_classes_gallery(self):
        return self._num_classes_gallery

    def get_num_classes(self, data):
        class_name_set = set()

        for datum in data:
            class_name_set.add(datum.class_name)

        return len(class_name_set)

    def download_data_from_gdrive(self, dst):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        gdown.download(self.data_url, dst, quiet=False)

        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(osp.dirname(dst))
        zip_ref.close()
        print("File Extracted to {}".format(osp.dirname(dst)))
