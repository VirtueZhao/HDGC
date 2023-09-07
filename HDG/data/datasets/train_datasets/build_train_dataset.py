from HDG.utils import Registry, check_availability

TRAIN_DATASET_REGISTRY = Registry("TRAIN_DATASET")


def build_train_dataset(cfg):
    available_datasets = TRAIN_DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.SOURCE_DOMAINS, available_datasets)
    print("Loading TRAIN Dataset: {}".format(cfg.DATASET.SOURCE_DOMAINS))

    return TRAIN_DATASET_REGISTRY.get(cfg.DATASET.SOURCE_DOMAINS)(cfg)
