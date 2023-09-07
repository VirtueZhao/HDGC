import torch
from tqdm import tqdm
from tabulate import tabulate
from collections import OrderedDict
from torch.nn import functional as F
from HDG.model import build_network
from HDG.engine.trainer import GenericNet
from HDG.utils import count_num_parameters, evaluator, TripletLoss
from HDG.engine import TRAINER_REGISTRY, GenericTrainer
from HDG.optim import build_optimizer, build_lr_scheduler


@TRAINER_REGISTRY.register()
class DDAIG(GenericTrainer):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA

    def build_model(self):
        print("Building Label Classifier")
        self.label_classifier = GenericNet(self.cfg, self.num_classes)
        self.label_classifier.to(self.device)
        self.optimizer_label = build_optimizer(self.label_classifier, self.cfg.OPTIM)
        self.scheduler_label = build_lr_scheduler(self.optimizer_label, self.cfg.OPTIM)
        self.model_registration("label_classifier", self.label_classifier, self.optimizer_label, self.scheduler_label)

        print("Building Domain Classifier")
        self.domain_classifier = GenericNet(self.cfg, self.num_source_domains)
        self.domain_classifier.to(self.device)
        self.optimizer_domain = build_optimizer(self.domain_classifier, self.cfg.OPTIM)
        self.scheduler_domain = build_lr_scheduler(self.optimizer_domain, self.cfg.OPTIM)
        self.model_registration("domain_classifier", self.domain_classifier, self.optimizer_domain, self.scheduler_domain)

        print("Building Domain Transformation Net")
        self.domain_transformation_net = build_network(self.cfg.TRAINER.DDAIG.G_ARCH)
        self.domain_transformation_net.to(self.device)
        self.optimizer_domain_transformation = build_optimizer(self.domain_transformation_net, self.cfg.OPTIM)
        self.scheduler_domain_transformation = build_lr_scheduler(self.optimizer_domain_transformation, self.cfg.OPTIM)
        self.model_registration("domain_transformation_net", self.domain_transformation_net, self.optimizer_domain_transformation, self.scheduler_domain_transformation)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"],
            ["Domain Transformation Net", f"{count_num_parameters(self.domain_transformation_net):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        #############
        # Update Domain Transformation Net
        #############
        input_data_p = self.domain_transformation_net(input_data, lmda=self.lmda)
        #
        loss_dt = 0
        # Minimize Label Classifier Loss
        loss_dt += F.cross_entropy(self.label_classifier(input_data_p), class_label)
        # # Maximize Domain Classifier Loss
        loss_dt -= F.cross_entropy(self.domain_classifier(input_data_p), domain_label)
        self.model_backward_and_update(loss_dt, "domain_transformation_net")

        # Perturb Data with Updated Domain Transformation Net
        with torch.no_grad():
            input_data_p = self.domain_transformation_net(input_data, lmda=self.lmda)

        #############
        # Update Label Classifier
        #############
        triplet_loss = TripletLoss(margin=1.2)
        output_original, representations_original = self.label_classifier(input_data, return_feature=True)
        output_perturbed, representations_perturbed = self.label_classifier(input_data_p, return_feature=True)

        loss_c1 = F.cross_entropy(output_original, class_label)
        loss_c2 = F.cross_entropy(output_perturbed, class_label)
        loss_t1 = triplet_loss(representations_original, class_label)
        loss_t2 = triplet_loss(representations_perturbed, class_label)
        loss_l = (1.0 - self.alpha) * (loss_c1 + loss_t1) + self.alpha * (loss_c2 + loss_t2)

        self.model_backward_and_update(loss_l, "label_classifier")

        #############
        # Update Domain Classifier
        #############
        loss_d = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        self.model_backward_and_update(loss_d, "domain_classifier")

        loss_summary = {
            # "loss_dt": loss_dt.item(),
            "loss_l": loss_l.item(),
            "loss_d": loss_d.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.label_classifier(input_data)

    def test(self):
        print("Extracting Feature Representation for Query Set and Gallery Set")
        self.set_model_mode("eval")

        representations = OrderedDict()
        class_names_labels = OrderedDict()

        with torch.no_grad():
            self.label_classifier.classifier = None
            
            for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
                file_names, input_data, class_names = self.parse_batch_test(batch_data)
                outputs = self.model_inference(input_data)
                outputs = outputs.cpu()

                for file_name, representation, class_name in zip(file_names, outputs, class_names):
                    representations[file_name] = representation
                    class_names_labels[file_name] = class_name

        dist_mat = evaluator.compute_dist_mat(representations, self.data_manager.test_dataset)
        evaluator.evaluate(dist_mat, self.data_manager.test_dataset)
