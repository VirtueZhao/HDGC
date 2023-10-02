import copy

import torch
from tqdm import tqdm
from tabulate import tabulate
from collections import OrderedDict
from HDG.model import build_network
from torch.nn import functional as F
from torchvision.transforms import Normalize
from HDG.engine.trainer import GenericNet, UnitClassifier
from HDG.engine import TRAINER_REGISTRY, GenericTrainer
from HDG.optim import build_optimizer, build_lr_scheduler
from HDG.utils import count_num_parameters, evaluator, TripletLoss, measure_diversity, compute_impact_factor


@TRAINER_REGISTRY.register()
class DCAIG(GenericTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda_domain = 0.2
        self.lmda_class = 0.2
        self.alpha = 0.5

    def build_model(self):
        print("Build Feature Extractor")
        self.feature_extractor = GenericNet(self.cfg, self.attribute_size)
        self.feature_extractor.to(self.device)
        self.optimizer_feature_extractor = build_optimizer(self.feature_extractor, self.cfg.OPTIM)
        self.scheduler_feature_extractor = build_lr_scheduler(self.optimizer_feature_extractor, self.cfg.OPTIM)
        # self.model_registration("feature_extractor", self.feature_extractor, self.optimizer_feature_extractor, self.scheduler_feature_extractor)
        self.feature_extractor_classifier = UnitClassifier(self.data_manager.dataset.attributes_dict, self.data_manager.dataset.seen)
        self.feature_extractor_classifier.to(self.device)
        self.feature_extractor_classifier.eval()

        print("Build Domain Generator")
        self.domain_generator = build_network("fcn_3x32_gctx")
        self.domain_generator.to(self.device)
        self.optimizer_domain_generator = build_optimizer(self.domain_generator, self.cfg.OPTIM)
        self.scheduler_domain_generator = build_lr_scheduler(self.optimizer_domain_generator, self.cfg.OPTIM)
        # self.model_registration("domain_generator", self.domain_generator, self.optimizer_domain_generator, self.scheduler_domain_generator)

        print("Build Domain Discriminator")
        self.domain_discriminator = GenericNet(self.cfg, self.num_source_domains)
        self.domain_discriminator.to(self.device)
        self.optimizer_domain_discriminator = build_optimizer(self.domain_discriminator, self.cfg.OPTIM)
        self.scheduler_domain_discriminator = build_lr_scheduler(self.optimizer_domain_discriminator, self.cfg.OPTIM)
        # self.model_registration("domain_discriminator", self.domain_discriminator, self.optimizer_domain_discriminator, self.scheduler_domain_discriminator)

        print("Build Class Generator")
        self.class_generator = build_network("fcn_3x32_gctx")
        self.class_generator.to(self.device)
        self.optimizer_class_generator = build_optimizer(self.class_generator, self.cfg.OPTIM)
        self.scheduler_class_generator = build_lr_scheduler(self.optimizer_class_generator, self.cfg.OPTIM)
        # self.model_registration("class_generator", self.class_generator, self.optimizer_class_generator, self.scheduler_class_generator)

        print("Build Class Discriminator")
        self.class_discriminator = GenericNet(self.cfg, self.attribute_size)
        self.class_discriminator.to(self.device)
        self.optimizer_class_discriminator = build_optimizer(self.class_discriminator, self.cfg.OPTIM)
        self.scheduler_class_discriminator = build_lr_scheduler(self.optimizer_class_discriminator, self.cfg.OPTIM)
        # self.model_registration("class_discriminator", self.class_discriminator, self.optimizer_class_discriminator, self.scheduler_class_discriminator)
        self.class_discriminator_classifier = UnitClassifier(self.data_manager.dataset.attributes_dict, self.data_manager.dataset.seen)
        self.class_discriminator_classifier.to(self.device)
        self.class_discriminator_classifier.eval()

        self.test_classifier = UnitClassifier(self.data_manager.dataset.attributes_dict, self.data_manager.dataset.unseen)
        self.test_classifier.to(self.device)
        self.test_classifier.eval()

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Feature Extractor", f"{count_num_parameters(self.feature_extractor):,}"],
            ["Domain Generator", f"{count_num_parameters(self.domain_generator):,}"],
            ["Domain Discriminator", f"{count_num_parameters(self.domain_discriminator):,}"],
            ["Class Generator", f"{count_num_parameters(self.class_generator):,}"],
            ["Class Discriminator", f"{count_num_parameters(self.class_discriminator):,}"]
        ]

        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        if self.current_epoch + 1 <= 5:
            # print("Warm-up Epoch")
            output, representations = self.feature_extractor(input_data, return_feature=True)
            c_loss = F.cross_entropy(output, class_label)
            triplet_loss = TripletLoss(margin=1.2)
            t_loss = triplet_loss(representations, class_label)
            loss_feature_extractor = c_loss + t_loss

            self.model_backward_and_update(loss_feature_extractor)

            loss_summary = {
                "loss_feature_extractor": loss_feature_extractor.item()
            }
        else:
            # Compute Diversity
            temp_data = copy.deepcopy(input_data)
            temp_model = copy.deepcopy(self.feature_extractor)
            _, embeddings = temp_model(temp_data, return_feature=True)
            embeddings_min = torch.min(embeddings)
            embeddings_max = torch.max(embeddings)
            normalized_embeddings = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)
            diversity = measure_diversity(normalized_embeddings.cpu(), diversity_type="gini")
            # print(type(diversity))
            # print(diversity.shape)
            # print("Current Diversity: {}".format(diversity.mean()))
            self.lmda_domain = self.lmda_class = compute_impact_factor(diversity, lower_bound=0, upper_bound=0.2)
            # print("Current Lambda: {}".format(self.lmda_domain))

            # Update D-GAN
            # Update Domain Generator
            temp_input = copy.deepcopy(input_data)
            input_data_domain_augmented = self.domain_generator(temp_input, lmda=self.lmda_domain)
            temp_feature_extractor = copy.deepcopy(self.feature_extractor)

            temp_semantic_projection = temp_feature_extractor(input_data_domain_augmented)
            temp_prediction = self.feature_extractor_classifier(temp_semantic_projection)
            loss_domain_generator = F.cross_entropy(temp_prediction, class_label)
            loss_domain_generator -= F.cross_entropy(self.domain_discriminator(input_data_domain_augmented), domain_label)
            self.domain_generator.zero_grad()
            self.detect_abnormal_loss(loss_domain_generator)
            loss_domain_generator.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.domain_generator.parameters(), max_norm=10, norm_type=2)
            self.optimizer_domain_generator.step()
            # self.model_backward_and_update(loss_domain_generator, "domain_generator")

            # Update Domain Discriminator
            input_data_domain_augmented, domain_perturbation = self.domain_generator(temp_input, lmda=self.lmda_domain, return_p=True)
            prediction = self.domain_discriminator(temp_input)
            loss_domain_discriminator_o = F.cross_entropy(prediction, domain_label)
            prediction_d = self.domain_discriminator(input_data_domain_augmented)
            loss_domain_discriminator_d = F.cross_entropy(prediction_d, domain_label)
            loss_domain_discriminator = loss_domain_discriminator_o + loss_domain_discriminator_d
            self.domain_discriminator.zero_grad()
            # self.detect_abnormal_loss(loss_domain_discriminator)

            if not torch.isfinite(loss_domain_discriminator).all():
                print("Start Debugging")
                print("Loss: {}".format(loss_domain_discriminator))
                print("Loss O: {}".format(loss_domain_discriminator_o))
                print("Loss D: {}".format(loss_domain_discriminator_d))
                print("-----")
                print("Prediction O: {}".format(prediction))
                print("Prediction D: {}".format(prediction_d))
                print("Class Label: {}".format(domain_label))
                print("-----")
                print("Original Input: {}".format(temp_input))
                print("Domain Augmented Input: {}".format(input_data_domain_augmented))
                nan_check = torch.isnan(input_data_domain_augmented)
                inf_check = torch.isinf(input_data_domain_augmented)
                has_nan = torch.any(nan_check)
                has_inf = torch.any(inf_check)
                print("Contains NaN: {}".format(has_nan.item()))
                print("Contains Inf: {}".format(has_inf.item()))
                print("-----")
                print("Domain Perturbation: {}".format(domain_perturbation))
                raise FloatingPointError("Loss is Infinite or NaN.")

            loss_domain_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.domain_discriminator.parameters(), max_norm=10, norm_type=2)
            self.optimizer_domain_discriminator.step()
            # self.model_backward_and_update(loss_domain_discriminator, "domain_discriminator")

            # Update C-GAN
            # Update Class Generator
            temp_input = copy.deepcopy(input_data)
            input_data_class_augmented = self.class_generator(temp_input, lmda=self.lmda_class)
            temp_feature_extractor = copy.deepcopy(self.feature_extractor)

            semantic_projection_class_augmented_f = temp_feature_extractor(input_data_class_augmented)
            prediction_class_augmented_f = self.feature_extractor_classifier(semantic_projection_class_augmented_f)
            loss_class_generator = F.cross_entropy(prediction_class_augmented_f, class_label)
            semantic_projection_class_augmented_c = self.class_discriminator(input_data_class_augmented)
            prediction_class_augmented_c = self.class_discriminator_classifier(semantic_projection_class_augmented_c)
            loss_class_generator -= F.cross_entropy(prediction_class_augmented_c, class_label)
            self.class_generator.zero_grad()
            self.detect_abnormal_loss(loss_class_generator)
            loss_class_generator.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.class_generator.parameters(), max_norm=10, norm_type=2)
            self.optimizer_class_generator.step()
            # self.model_backward_and_update(loss_class_generator, "class_generator")

            # Update Class Discriminator
            input_data_class_augmented, class_perturbation = self.class_generator(input_data, lmda=self.lmda_class, return_p=True)
            semantic_projection = self.class_discriminator(input_data)
            prediction = self.class_discriminator_classifier(semantic_projection)
            loss_class_discriminator_o = F.cross_entropy(prediction, class_label)

            semantic_projection_c = self.class_discriminator(input_data_class_augmented)
            prediction_c = self.class_discriminator_classifier(semantic_projection_c)
            loss_class_discriminator_c = F.cross_entropy(prediction_c, class_label)

            loss_class_discriminator = loss_class_discriminator_o + loss_class_discriminator_c
            self.class_discriminator.zero_grad()
            # self.detect_abnormal_loss(loss_class_discriminator)

            if not torch.isfinite(loss_class_discriminator).all():
                print("Start Debugging")
                # print("Loss: {}".format(loss_class_discriminator))
                # print("Loss O: {}".format(loss_class_discriminator_o))
                # print("Loss C: {}".format(loss_class_discriminator_c))
                # print("-----")
                # print("Prediction O: {}".format(prediction))
                # print("Prediction C: {}".format(prediction_c))
                # print("Class Label: {}".format(class_label))
                # print("-----")
                print("Original Input: {}".format(temp_input))
                print("Class Augmented Input: {}".format(input_data_class_augmented))
                nan_check = torch.isnan(input_data_class_augmented)
                inf_check = torch.isinf(input_data_class_augmented)
                has_nan = torch.any(nan_check)
                has_inf = torch.any(inf_check)
                print("Contains NaN: {}".format(has_nan.item()))
                print("Contains Inf: {}".format(has_inf.item()))
                print("-----")
                print("Class Perturbation: {}".format(class_perturbation))
                raise FloatingPointError("Loss is Infinite or NaN.")

            loss_class_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.class_discriminator.parameters(), max_norm=10, norm_type=2)
            self.optimizer_class_discriminator.step()
            # self.model_backward_and_update(loss_class_discriminator, "class_discriminator")

            # Update Feature Extractor
            # Generate Domain Perturbation and Class Perturbation
            with torch.no_grad():
                temp_input = copy.deepcopy(input_data)
                _, domain_perturbation = self.domain_generator(temp_input, lmda=self.lmda_domain, return_p=True)
                _, class_perturbation = self.class_generator(temp_input, lmda=self.lmda_class, return_p=True)
                # self.lmda_domain = self.lmda_class = 0
                input_data_domain_class_augmented = temp_input + self.lmda_domain * domain_perturbation + self.lmda_class * class_perturbation

            semantic_projection, representations = self.feature_extractor(input_data, return_feature=True)
            semantic_projection_augmented, representations_augmented = self.feature_extractor(input_data_domain_class_augmented, return_feature=True)

            prediction = self.feature_extractor_classifier(semantic_projection)
            prediction_augmented = self.feature_extractor_classifier(semantic_projection_augmented)
            loss_c1 = F.cross_entropy(prediction, class_label)
            loss_c2 = F.cross_entropy(prediction_augmented, class_label)
            triplet_loss = TripletLoss(margin=1.2)
            loss_t1 = triplet_loss(representations, class_label)
            loss_t2 = triplet_loss(representations_augmented, class_label)
            loss_feature_extractor = (1.0 - self.alpha) * (loss_c1 + loss_t1) + self.alpha * (loss_c2 + loss_t2)
            self.feature_extractor.zero_grad()
            self.detect_abnormal_loss(loss_feature_extractor)
            loss_feature_extractor.backward()
            self.optimizer_feature_extractor.step()
            # self.model_backward_and_update(loss_feature_extractor, "feature_extractor")

            loss_summary = {
                "loss_feature_extractor": loss_feature_extractor.item(),
                "loss_domain_generator": loss_domain_generator.item(),
                "loss_domain_discriminator": loss_domain_discriminator.item(),
                "loss_class_generator": loss_class_generator.item(),
                "loss_class_discriminator": loss_class_discriminator.item()
            }

        if self.batch_index + 1 == self.num_batches:
            self.scheduler_feature_extractor.step()
            self.scheduler_domain_generator.step()
            self.scheduler_domain_discriminator.step()
            self.scheduler_class_generator.step()
            self.scheduler_class_discriminator.step()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.feature_extractor(input_data)

    def get_current_lr(self):
        return self.optimizer_feature_extractor.param_groups[0]["lr"]

    # def test(self):
    #     print("Extracting Feature Representation for Query Set and Gallery Set")
    #     self.set_model_mode("eval")
    #
    #     representations = OrderedDict()
    #     class_names_labels = OrderedDict()
    #
    #     with torch.no_grad():
    #         self.feature_extractor.semantic_projector = None
    #
    #         for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
    #             file_names, input_data, class_names = self.parse_batch_test(batch_data)
    #
    #             outputs = self.model_inference(input_data)
    #             outputs = outputs.cpu()
    #
    #             for file_name, representation, class_name in zip(file_names, outputs, class_names):
    #                 representations[file_name] = representation
    #                 class_names_labels[file_name] = class_name
    #
    #         dist_mat = evaluator.compute_dist_mat(representations, self.data_manager.test_dataset)
    #         evaluator.evaluate(dist_mat, self.data_manager.test_dataset)

    def test(self):
        self.feature_extractor.eval()
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(self.test_data_loader)):
                input_data, class_label = self.parse_batch_test(batch_data)
                semantic_projection = self.feature_extractor(input_data)
                prediction = self.test_classifier(semantic_projection)
                self.evaluator.process(prediction, class_label)
            evaluation_results = self.evaluator.evaluate()

            for k, v in evaluation_results.items():
                self.write_scalar(f"test/{k}", v, self.current_epoch)

            return list(evaluation_results.values())[0]
