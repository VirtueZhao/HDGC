import time
import torch
import datetime
import os.path as osp
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate
from collections import OrderedDict
from HDG.data import DataManager
from HDG.model import build_backbone
from HDG.optim import build_optimizer, build_lr_scheduler
from HDG.utils import tolist_if_not, count_num_parameters, mkdir_if_missing, MetricMeter, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from HDG.evaluation import build_evaluator

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class UnitClassifier(nn.Module):
    def __init__(self, attributes, classes):
        super(UnitClassifier, self).__init__()

        self.fc = nn.Linear(attributes[0].size(0), classes.size(0), bias=False)

        for index, cls in enumerate(classes):
            norm_attributes = attributes[cls.item()]
            norm_attributes /= torch.norm(norm_attributes, 2)
            self.fc.weight[index].data[:] = norm_attributes

    def forward(self, x):
        return self.fc(x)


class GenericNet(nn.Module):
    """A generic neural network that composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, attribute_size, **kwargs):
        super().__init__()
        self.backbone = build_backbone(cfg, **kwargs)
        self._out_features = self.backbone.out_features
        self.semantic_projector = None
        self.semantic_projector = nn.Linear(self._out_features, attribute_size)

    @property
    def out_features(self):
        return self._out_features

    def forward(self, x, return_feature=False):
        f = self.backbone(x)

        if self.semantic_projector is None:
            return f

        y = self.semantic_projector(f)

        if return_feature:
            return y, f

        return y


class BaseTrainer:
    """Base Class for Iterative Trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optimizers = OrderedDict()
        self._schedulers = OrderedDict()
        self._writer = None

    def model_registration(self, model_name="model", model=None, optimizer=None, scheduler=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before super().__init__() call")
        if self.__dict__.get("_optimizers") is None:
            raise AttributeError("Cannot assign optimizer before super().__init__() call")
        if self.__dict__.get("_schedulers") is None:
            raise AttributeError("Cannot assign scheduler before super().__init__() call")

        assert model_name not in self._models, "Found duplicate model names"

        self._models[model_name] = model
        self._optimizers[model_name] = optimizer
        self._schedulers[model_name] = scheduler

    def get_model_names(self, model_names=None):
        model_names_real = list(self._models.keys())
        if model_names is not None:
            model_names = tolist_if_not(model_names)
            for model_name in model_names:
                assert model_name in model_names_real
            return model_names
        else:
            return model_names_real

    def save_model(self, epoch, directory):
        model_names = self.get_model_names()

        for name in model_names:
            model_state_dict = self._models[name].state_dict()

            optimizer_state_dict = None
            if self._optimizers[name] is not None:
                optimizer_state_dict = self._optimizers[name].state_dict()

            scheduler_state_dict = None
            if self._schedulers[name] is not None:
                scheduler_state_dict = self._schedulers[name].state_dict()

            fpath = osp.join(directory, name)
            mkdir_if_missing(fpath)
            model_name = "model.pth.tar-" + str(epoch + 1)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                    "scheduler_state_dict": scheduler_state_dict
                },
                osp.join(fpath, model_name)
            )
            print("Model Saved to: {}".format(osp.join(fpath, model_name)))

    def set_model_mode(self, mode="train", model_names=None):
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if mode == "train":
                self._models[model_name].train()
            elif mode in ["test", "eval"]:
                self._models[model_name].eval()
            else:
                raise KeyError

    def update_lr(self, model_names=None):
        model_names = self.get_model_names(model_names)

        for model_name in model_names:
            if self._schedulers[model_name] is not None:
                self._schedulers[model_name].step()

    def detect_abnormal_loss(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is Infinite or NaN.")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            # print("Initializing Summary Writer with log_dir={}".format(log_dir))
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic Training Loops"""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # self.before_train()
        # for self.current_epoch in range(self.start_epoch, self.max_epoch):
        #     self.before_epoch()
        #     self.run_epoch()
        #     self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch_data):
        raise NotImplementedError

    def parse_batch_test(self, batch_data):
        raise NotImplementedError

    def forward_backward(self, batch_data):
        raise NotImplementedError

    def model_zero_grad(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].zero_grad()

    def model_backward(self, loss):
        self.detect_abnormal_loss(loss)
        loss.backward()

    def model_update_optimizer(self, model_names=None):
        model_names = self.get_model_names(model_names)
        for model_name in model_names:
            if self._optimizers[model_name] is not None:
                self._optimizers[model_name].step()

    def model_backward_and_update(self, loss, model_names=None):
        self.model_zero_grad(model_names)
        self.model_backward(loss)
        self.model_update_optimizer(model_names)


class GenericTrainer(BaseTrainer):
    """Generic Trainer Class for Implementing Generic Function"""

    def __init__(self, cfg):
        super().__init__()

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device(("cuda:{}".format(cfg.GPU)))
        else:
            self.device = torch.device("cpu")

        self.start_epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg

        # Build Data Loader
        self.data_manager = DataManager(self.cfg)
        self.train_data_loader = self.data_manager.train_data_loader
        self.test_data_loader = self.data_manager.test_data_loader

        self.attribute_size = self.data_manager.attribute_size
        self.num_classes_train = self.data_manager.num_classes_train
        self.num_classes_test = self.data_manager.num_classes_test
        self.num_source_domains = self.data_manager.num_source_domains

        self.build_model()
        self.evaluator = build_evaluator(self.cfg)

    def build_model(self):
        """Build and Register Default Model.

        Custom Trainers Can Re-Implement This Method If Necessary.
        """

        self.model = GenericNet(self.cfg, self.attribute_size)
        self.model.to(self.device)
        model_parameters_table = [
            ["Model", "# Parameters"],
            ["ERM", f"{count_num_parameters(self.model):,}"]
        ]
        print(tabulate(model_parameters_table))

        self.optimizer = build_optimizer(self.model, self.cfg.OPTIM)
        self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
        self.model_registration("model", self.model, self.optimizer, self.scheduler)

        self.train_classifier = UnitClassifier(self.data_manager.dataset.attributes_dict, self.data_manager.dataset.seen)
        self.train_classifier.to(self.device)
        self.train_classifier.eval()
        self.test_classifier = UnitClassifier(self.data_manager.dataset.attributes_dict, self.data_manager.dataset.unseen)
        self.test_classifier.to(self.device)
        self.test_classifier.eval()

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        # Initialize SummaryWriter
        # writer_dir = osp.join(self.output_dir, "tensorboard")
        # mkdir_if_missing(writer_dir)
        # self.init_writer(writer_dir)

        self.time_start = time.time()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_data_loader)

        end_time = time.time()

        for self.batch_index, batch_data in enumerate(self.train_data_loader):
            data_time.update(time.time() - end_time)
            loss_summary = self.forward_backward(batch_data)
            batch_time.update(time.time() - end_time)
            losses.update(loss_summary)

            if (self.batch_index + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                num_batches_remain = 0
                num_batches_remain += self.num_batches - self.batch_index - 1
                num_batches_remain += (self.max_epoch - self.current_epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * num_batches_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.current_epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_index + 1}/{self.num_batches}]"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            # n_iter = self.current_epoch * self.num_batches + self.batch_index
            # for name, meter in losses.meters.items():
            #     self.write_scalar("train/" + name, meter.avg, n_iter)
            # self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end_time = time.time()

    def after_train(self):
        print("Finish Training")
        # self.save_model(self.current_epoch, self.output_dir)
        self.test()

    # def test(self):
    #     print("Extracting Feature Representation for Query Set and Gallery Set")
    #     self.set_model_mode("eval")
    #
    #     representations = OrderedDict()
    #     class_names_labels = OrderedDict()
    #
    #     with torch.no_grad():
    #         self.model.semantic_projector = None
    #
    #         for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
    #             file_names, input_data, class_names = self.parse_batch_test(batch_data)
    #             outputs = self.model_inference(input_data)
    #             outputs = outputs.cpu()
    #
    #             for file_name, representation, class_name in zip(file_names, outputs, class_names):
    #                 representations[file_name] = representation
    #                 class_names_labels[file_name] = class_name
    #
    #     dist_mat = evaluator.compute_dist_mat(representations, self.data_manager.test_dataset)
    #     evaluator.evaluate(dist_mat, self.data_manager.test_dataset)

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
                input_data, class_label = self.parse_batch_test(batch_data)
                semantic_projection = self.model(input_data)
                prediction = self.test_classifier(semantic_projection)

                print(self.evaluator)
                exit()


    def model_inference(self, input_data):
        return self.model(input_data)

    # def parse_batch_test(self, batch_data):
    #     file_name = [path.split('/')[-1] for path in batch_data["img_path"]]
    #     input_data = batch_data["img"].to(self.device)
    #     class_name = batch_data["class_name"]
    #     class_name = torch.tensor(list(map(int, class_name)), dtype=torch.int64)
    #
    #     return file_name, input_data, class_name

    def parse_batch_test(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)

        return input_data, class_label

    def get_current_lr(self, names=None):
        name = self.get_model_names(names)[0]
        return self._optimizers[name].param_groups[0]["lr"]
