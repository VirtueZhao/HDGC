import torch
from tqdm import tqdm
from tabulate import tabulate
from collections import OrderedDict
from torch.nn import functional as F
from HDG.engine.trainer import GenericNet
from HDG.utils import count_num_parameters, evaluator, TripletLoss
from HDG.engine import TRAINER_REGISTRY, GenericTrainer
from HDG.optim import build_optimizer, build_lr_scheduler

@TRAINER_REGISTRY.register()
class CrossGrad(GenericTrainer):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_l = cfg.TRAINER.CROSSGRAD.EPS_L
        self.eps_d = cfg.TRAINER.CROSSGRAD.EPS_D
        self.alpha_l = cfg.TRAINER.CROSSGRAD.ALPHA_L
        self.alpha_d = cfg.TRAINER.CROSSGRAD.ALPHA_D

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

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        input_data.requires_grad = True

        # Compute Domain Perturbation
        loss_domain = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_domain.backward()
        grad_domain = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_domain_perturb = input_data.data + self.eps_l * grad_domain

        # Compute Label Perturbation
        input_data.grad.data.zero_()
        loss_label = F.cross_entropy(self.label_classifier(input_data), class_label)
        loss_label.backward()
        grad_label = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_label_perturb = input_data.data + self.eps_d * grad_label

        input_data = input_data.detach()

        # Update Label Classifier
        triplet_loss = TripletLoss(margin=1.2)
        output_original, representations_original = self.label_classifier(input_data, return_feature=True)
        output_domain_perturb, representations_domain_perturb = self.label_classifier(input_data_domain_perturb, return_feature=True)

        loss_c1 = F.cross_entropy(output_original, class_label)
        loss_c2 = F.cross_entropy(output_domain_perturb, class_label)
        loss_t1 = triplet_loss(representations_original, class_label)
        loss_t2 = triplet_loss(representations_domain_perturb, class_label)
        loss_l = (1 - self.alpha_l) * (loss_c1 + loss_t1) + self.alpha_l * (loss_c2 + loss_t2)
        self.model_backward_and_update(loss_l, "label_classifier")

        # Update Domain Classifier
        loss_d1 = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_d2 = F.cross_entropy(self.domain_classifier(input_data_label_perturb), domain_label)
        loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
        self.model_backward_and_update(loss_d, "domain_classifier")

        loss_summary = {
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
            self.label_classifier.semantic_projector = None
            for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
                file_names, input_data, class_names = self.parse_batch_test(batch_data)
                outputs = self.model_inference(input_data)
                outputs = outputs.cpu()

                for file_name, representation, class_name in zip(file_names, outputs, class_names):
                    representations[file_name] = representation
                    class_names_labels[file_name] = class_name

        dist_mat = evaluator.compute_dist_mat(representations, self.data_manager.test_dataset)
        evaluator.evaluate(dist_mat, self.data_manager.test_dataset)
