import csv
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="submission",
)
def main(config):
    set_random_seed(config.submission.seed)

    if config.submission.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.submission.device

    print("Device:", device)

    dataloaders, batch_transforms = get_dataloaders(
        config,
        device,
    )

    assert "eval" in dataloaders, "Eval DataLoader не создан"
    eval_loader = dataloaders["eval"]

    model = instantiate(config.model).to(device)

    checkpoint_path = Path(config.submission.checkpoint)
    assert checkpoint_path.exists(), (
        f"Checkpoint не найден: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    inference_transforms = batch_transforms.get("inference")

    rows = []

    with torch.no_grad():
        for batch in tqdm(
            eval_loader,
            desc="eval submission",
            total=len(eval_loader),
        ):
            utterance_ids = list(batch["utterance_id"])

            # Переносим на GPU только Tensor-поля.
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(
                        device,
                        non_blocking=True,
                    )

            # Используем тот же STFT frontend, что и на dev.
            if inference_transforms is not None:
                for tensor_name in inference_transforms.keys():
                    batch[tensor_name] = (
                        inference_transforms[tensor_name](
                            batch[tensor_name]
                        )
                    )

            outputs = model(**batch)
            logits = outputs["logits"]

            # label 0 = spoof, label 1 = bonafide.
            # Чем выше score, тем сильнее поддержка bonafide.
            scores = (
                logits[:, 1] - logits[:, 0]
            ).detach().cpu().tolist()

            if len(utterance_ids) != len(scores):
                raise RuntimeError(
                    "Количество utterance_id и scores не совпало"
                )

            rows.extend(zip(utterance_ids, scores))

    utterance_ids = [utterance_id for utterance_id, _ in rows]

    if len(utterance_ids) != len(set(utterance_ids)):
        raise RuntimeError(
            "В предсказаниях найдены повторяющиеся utterance_id"
        )

    if len(rows) != len(eval_loader.dataset):
        raise RuntimeError(
            f"Получено {len(rows)} строк, "
            f"ожидалось {len(eval_loader.dataset)}"
        )

    output_path = Path(config.submission.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)

        # Заголовок не добавляем.
        for utterance_id, score in rows:
            writer.writerow([
                utterance_id,
                f"{score:.10f}",
            ])

    print("Submission saved:", output_path)
    print("Number of rows:", len(rows))
    print("First row:", rows[0])
    print("Last row:", rows[-1])


if __name__ == "__main__":
    main()
