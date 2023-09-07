from HDG.utils import Registry, check_availability

TEST_DATASET_REGISTRY = Registry("TEST_DATASET")


def build_test_dataset(cfg):
    available_datasets = TEST_DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.TARGET_DOMAIN, available_datasets)
    print("Loading TEST  Dataset: {}".format(cfg.DATASET.TARGET_DOMAIN))

    return TEST_DATASET_REGISTRY.get(cfg.DATASET.TARGET_DOMAIN)(cfg)
