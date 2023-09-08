import numpy as np
from torch.nn import functional as F
from HDG.engine import TRAINER_REGISTRY, GenericTrainer
from HDG.utils import TripletLoss, compute_top_k_accuracy

import matplotlib.pyplot as plt


@TRAINER_REGISTRY.register()
class ERM(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        input_data, class_label = self.parse_batch_train(batch_data)
        semantic_projection = self.model(input_data)
        prediction = self.train_classifier(semantic_projection)
        semantic_loss = F.cross_entropy(prediction, class_label)

        # triplet_loss = TripletLoss(margin=1.2)
        # t_loss = triplet_loss(semantic_projection, class_label)
        # loss = semantic_loss + t_loss
        loss = semantic_loss

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(prediction, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)

        return input_data, class_label
