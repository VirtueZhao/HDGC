import sys
import glob
import os.path as osp
from .base_test_dataset import Datum, BaseTestDataset
from .build_test_dataset import TEST_DATASET_REGISTRY


@TEST_DATASET_REGISTRY.register()
class MSMT17(BaseTestDataset):

    def __init__(self, cfg):
        self._dataset_dir = "MSMT17"
        self._data_url = "https://drive.google.com/uc?id=1k9sIN0O6f2MMrTFQxPQxsEzsq7qt1BFH"
        dataset_path = osp.abspath(osp.expanduser(cfg.DATASET.PATH))
        self._dataset_dir = osp.join(dataset_path, self._dataset_dir)
        self.test_info = {}

        if not osp.exists(self._dataset_dir):
            dst = osp.join(dataset_path, "MSMT17.zip")
            self.download_data_from_gdrive(dst)

        query_data, gallery_data = self.read_test_data()
        super().__init__(dataset_dir=self._dataset_dir, data_url=self._data_url, query_data=query_data, gallery_data=gallery_data)

    def read_test_data(self):

        def _build_datums(images):
            datums = []
            for img_path in images:
                if sys.platform == "linux":
                    file_name = img_path.split("/")[-1]
                else:
                    file_name = img_path.split("\\")[-1]

                class_name = file_name.split("_")[0]
                domain_label = int(file_name.split("_")[2])

                datum = Datum(
                    img_path=img_path,
                    class_name=class_name,
                    class_label=-1,
                    domain_label=domain_label
                )
                datums.append(datum)

            return datums

        query_dir = osp.join(self._dataset_dir, "Query")
        query_images = glob.glob(osp.join(query_dir, "*.jpg"))
        self.test_info["Query"] = len(query_images)

        gallery_dir = osp.join(self._dataset_dir, "Gallery")
        gallery_images = glob.glob(osp.join(gallery_dir, "*.jpg"))
        self.test_info["Gallery"] = len(gallery_images)

        query_datums = _build_datums(query_images)
        gallery_datums = _build_datums(gallery_images)

        return query_datums, gallery_datums
