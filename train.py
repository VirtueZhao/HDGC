import torch
import argparse
from HDG.utils import get_cfg_default, set_random_seed, setup_logger
from HDG.engine import build_trainer


def setup_cfg(args):
    cfg = get_cfg_default()

    reset_cfg_from_args(cfg, args)

    clean_cfg(cfg, args.trainer)

    cfg.freeze()

    return cfg


def reset_cfg_from_args(cfg, args):
    if args.gpu:
        cfg.GPU = args.gpu

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.img_height and args.img_width:
        cfg.INPUT.SIZE = (args.img_height, args.img_width)

    if args.dataset_path:
        cfg.DATASET.PATH = args.dataset_path

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domain:
        cfg.DATASET.TARGET_DOMAIN = args.target_domain

    if args.batch_size:
        cfg.DATALOADER.TRAIN.BATCH_SIZE = args.batch_size

    if args.batch_identity_size:
        cfg.DATALOADER.TRAIN.BATCH_IDENTITY_SIZE = args.batch_identity_size

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.max_epoch:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    cfg.DATASET.NAME = "DomainNet"

def clean_cfg(cfg, trainer):
    """Remove unused trainers (configs).

    Aim: Only show relevant information when calling print(cfg).

    Args:
        cfg (_C): cfg instance.
        trainer (str): baseline name.
    """
    keys = list(cfg.TRAINER.keys())
    for key in keys:
        if key == "NAME" or key == trainer.upper():
            continue
        cfg.TRAINER.pop(key, None)


def main(args):
    torch.set_num_threads(1)

    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print("** Config **")
    print(cfg)

    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="specify GPU"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="path to datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--max_epoch",
        default=1,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int
    )
    parser.add_argument(
        "--batch_identity_size",
        type=int,
        default=4,
        help="number of examples for each identity"
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float
    )
    parser.add_argument(
        "--source_domains",
        type=str,
        nargs="+",
        help="source domain for domain generalization"
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        help="target domain for domain generalization"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        help="name of trainers"
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=224,
        help="height of input images"
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=224,
        help="width of input images"
    )

    args = parser.parse_args()
    main(args)
