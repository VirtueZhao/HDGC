import math
from torch.nn import functional as F
from HDG.model.ops import SimpleAugOP
from HDG.engine import TRAINER_REGISTRY, GenericTrainer
from HDG.utils import TripletLoss, compute_top_k_accuracy


@TRAINER_REGISTRY.register()
class SimpleAug(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        input_data_original, class_label = self.parse_batch_train(batch_data)

        simple_aug = SimpleAugOP(aug_type="ColorJitter")
        input_data_augmented = simple_aug(input_data_original)
        output_original, representations_original = self.model(input_data_original, return_feature=True)
        output_augmented, representations_augmented = self.model(input_data_augmented, return_feature=True)

        c_loss_original = F.cross_entropy(output_original, class_label)
        c_loss_augmented = F.cross_entropy(output_augmented, class_label)

        triplet_loss = TripletLoss(margin=1.2)
        t_loss_original = triplet_loss(representations_original, class_label)
        t_loss_augmented = triplet_loss(representations_augmented, class_label)

        loss_original = c_loss_original + t_loss_original
        loss_augmented = c_loss_augmented + t_loss_augmented

        loss = 0.5 * loss_original + 0.5 * loss_augmented

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output_original, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        return input_data, class_label
