import torch
from tqdm.auto import tqdm

from src.metrics import compute_bonafide_scores, compute_eer
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run one batch through the model and loss function.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]

        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(
                loss_name,
                batch[loss_name].item(),
            )

        for metric in metric_funcs:
            metrics.update(
                metric.name,
                metric(**batch),
            )

        return batch

    def _evaluation_epoch(
        self,
        epoch,
        part,
        dataloader,
    ):
        """
        Evaluate the model and compute EER over the full partition.
        """

        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )

                scores = compute_bonafide_scores(
                    batch["logits"]
                )

                all_scores.append(
                    scores.detach().cpu()
                )

                all_labels.append(
                    batch["labels"].detach().cpu()
                )

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        eer, _ = compute_eer(
            bonafide_scores=bonafide_scores,
            spoof_scores=spoof_scores,
        )

        self.writer.set_step(
            epoch * self.epoch_len,
            part,
        )

        self._log_scalars(
            self.evaluation_metrics
        )

        self.writer.add_scalar(
            "EER",
            eer,
        )

        self._log_batch(
            batch_idx,
            batch,
            part,
        )

        logs = self.evaluation_metrics.result()
        logs["EER"] = eer

        return logs

    def _log_batch(
        self,
        batch_idx,
        batch,
        mode="train",
    ):
        """
        Log additional batch data if needed.
        """

        if mode == "train":
            pass
        else:
            pass