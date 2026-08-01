#!/usr/bin/env python3
"""Find and evaluate LCNN checkpoints on ASVspoof 2019 LA eval.

Run this file from the repository root or copy it there. It uses the
repository's existing model, dataset, transforms, loss, and EER code.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
EPOCH_PATTERNS = (
    re.compile(r"checkpoint[-_]?epoch[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"epoch[-_]?(\d+)", re.IGNORECASE),
)


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    epoch: int | None
    architecture: str | None
    monitor_best: float | None
    size_mb: float
    modified_at: str
    load_error: str | None = None


def repository_root() -> Path:
    """Find the repository root containing train.py and src/configs."""
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]

    for candidate in candidates:
        if (candidate / "train.py").exists() and (
            candidate / "src" / "configs"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "Не найден корень репозитория. Поместите скрипт в каталог "
        "Voice-Anti-spoofing-LCNN или запускайте его из этого каталога."
    )


def parse_epoch_from_name(path: Path) -> int | None:
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(path.stem)
        if match:
            return int(match.group(1))
    return None


def inspect_checkpoint(path: Path) -> CheckpointInfo:
    epoch: int | None = None
    architecture: str | None = None
    monitor_best: float | None = None
    load_error: str | None = None

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            raw_epoch = checkpoint.get("epoch")
            if raw_epoch is not None:
                epoch = int(raw_epoch)

            raw_architecture = checkpoint.get("arch")
            if raw_architecture is not None:
                architecture = str(raw_architecture)

            raw_monitor_best = checkpoint.get("monitor_best")
            if raw_monitor_best is not None:
                monitor_best = float(raw_monitor_best)

        if epoch is None:
            epoch = parse_epoch_from_name(path)

    except Exception as error:  # noqa: BLE001 - inventory must continue
        load_error = f"{type(error).__name__}: {error}"
        epoch = parse_epoch_from_name(path)

    stat = path.stat()
    return CheckpointInfo(
        path=str(path.resolve()),
        epoch=epoch,
        architecture=architecture,
        monitor_best=monitor_best,
        size_mb=stat.st_size / 1024**2,
        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        ),
        load_error=load_error,
    )


def normalize_roots(roots: Sequence[str]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[Path] = set()

    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            print(f"Пропущен отсутствующий каталог: {root}")
            continue
        if root in seen:
            continue
        seen.add(root)
        normalized.append(root)

    return normalized


def iter_checkpoint_paths(roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()

    for root in roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
            )

        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def discover_checkpoints(roots: Sequence[Path]) -> list[CheckpointInfo]:
    paths = sorted(iter_checkpoint_paths(roots))
    inventory: list[CheckpointInfo] = []

    for path in tqdm(paths, desc="Inspecting checkpoints"):
        inventory.append(inspect_checkpoint(path))

    inventory.sort(
        key=lambda item: (
            item.epoch is None,
            item.epoch if item.epoch is not None else 10**9,
            item.path,
        )
    )
    return inventory


def inventory_dataframe(inventory: Sequence[CheckpointInfo]) -> pd.DataFrame:
    columns = [
        "epoch",
        "architecture",
        "monitor_best",
        "size_mb",
        "modified_at",
        "path",
        "load_error",
    ]
    rows = [asdict(item) for item in inventory]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def print_inventory(inventory: Sequence[CheckpointInfo]) -> None:
    if not inventory:
        print("Checkpoint-файлы не найдены.")
        return

    table = inventory_dataframe(inventory).copy()
    table["size_mb"] = table["size_mb"].map(lambda value: f"{value:.2f}")
    table["monitor_best"] = table["monitor_best"].map(
        lambda value: "" if pd.isna(value) else f"{value:.8f}"
    )
    table["load_error"] = table["load_error"].fillna("")
    print(table.to_string(index=False))


def save_inventory(
    inventory: Sequence[CheckpointInfo],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_dataframe(inventory).to_csv(output_path, index=False)
    print(f"Инвентаризация сохранена: {output_path}")


def choose_checkpoints_by_epoch(
    inventory: Sequence[CheckpointInfo],
    epochs: Sequence[int],
) -> list[Path]:
    selected: list[Path] = []

    for epoch in epochs:
        candidates = [
            Path(item.path)
            for item in inventory
            if item.epoch == epoch and item.load_error is None
        ]

        if not candidates:
            raise FileNotFoundError(
                f"Не найден читаемый checkpoint эпохи {epoch}. "
                "Сначала выполните команду list и проверьте пути."
            )

        if len(candidates) > 1:
            candidate_text = "\n".join(f"  {path}" for path in candidates)
            raise RuntimeError(
                f"Найдено несколько checkpoint-файлов эпохи {epoch}:\n"
                f"{candidate_text}\n"
                "Передайте нужные файлы явно через --checkpoint."
            )

        selected.append(candidates[0])

    return selected


def load_checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        return int(checkpoint["epoch"])

    epoch = parse_epoch_from_name(path)
    if epoch is None:
        raise ValueError(
            f"Не удалось определить эпоху checkpoint-файла: {path}"
        )
    return epoch


def build_evaluation_config(
    project_root: Path,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
):
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as error:
        raise RuntimeError(
            "Hydra не установлена. Выполните: pip install hydra-core"
        ) from error

    protocol_path = (
        dataset_root
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.eval.trl.txt"
    )
    audio_dir = dataset_root / "ASVspoof2019_LA_eval" / "flac"

    if not protocol_path.exists():
        raise FileNotFoundError(f"Не найден eval-протокол: {protocol_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Не найдены eval-аудиозаписи: {audio_dir}")

    config_dir = project_root / "src" / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Не найден каталог Hydra-конфигураций: {config_dir}"
        )

    overrides = [
        f"datasets.eval.protocol_path={protocol_path}",
        f"datasets.eval.audio_dir={audio_dir}",
        f"dataloader.batch_size={batch_size}",
        f"dataloader.num_workers={num_workers}",
    ]

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        return compose(
            config_name="submission",
            overrides=overrides,
        )


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Запрошен CUDA, но GPU недоступен.")
    return requested_device


def apply_inference_transforms(batch: dict, transforms) -> dict:
    if transforms is None:
        return batch

    for tensor_name, transform in transforms.items():
        batch[tensor_name] = transform(batch[tensor_name])
    return batch


def load_state_dict_into_model(model, checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Неподдерживаемый формат checkpoint: {checkpoint_path}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def evaluate_one_checkpoint(
    *,
    checkpoint_path: Path,
    model,
    criterion,
    eval_loader,
    inference_transforms,
    device: str,
) -> dict:
    from src.metrics import compute_bonafide_scores, compute_eer

    checkpoint = load_state_dict_into_model(
        model,
        checkpoint_path,
        device,
    )
    epoch = int(checkpoint.get("epoch", load_checkpoint_epoch(checkpoint_path)))

    model.eval()
    total_loss = 0.0
    total_objects = 0
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in tqdm(
            eval_loader,
            desc=f"Eval epoch {epoch}",
            total=len(eval_loader),
        ):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)

            batch = apply_inference_transforms(
                batch,
                inference_transforms,
            )

            outputs = model(**batch)
            logits = outputs["logits"]
            labels = batch["labels"]

            loss = criterion(
                logits=logits,
                labels=labels,
            )["loss"]

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_objects += batch_size

            all_scores.append(
                compute_bonafide_scores(logits).detach().cpu()
            )
            all_labels.append(labels.detach().cpu())

    if total_objects != len(eval_loader.dataset):
        raise RuntimeError(
            f"Обработано {total_objects} объектов, "
            f"ожидалось {len(eval_loader.dataset)}."
        )

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    eer, threshold = compute_eer(
        bonafide_scores=bonafide_scores,
        spoof_scores=spoof_scores,
    )

    result = {
        "checkpoint_epoch": epoch,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "eval_loss": total_loss / total_objects,
        "eval_eer": eer,
        "eval_eer_percent": eer * 100.0,
        "eer_threshold": threshold,
        "number_of_objects": total_objects,
        "bonafide_objects": int((labels == 1).sum()),
        "spoof_objects": int((labels == 0).sum()),
    }

    print(
        f"Epoch {epoch}: "
        f"eval_loss={result['eval_loss']:.8f}, "
        f"eval_EER={result['eval_eer_percent']:.4f} %, "
        f"objects={total_objects}"
    )
    return result


def save_results_csv(results: Sequence[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_epoch",
        "eval_loss",
        "eval_eer",
        "eval_eer_percent",
        "eer_threshold",
        "number_of_objects",
        "bonafide_objects",
        "spoof_objects",
        "checkpoint_path",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Результаты сохранены: {output_path}")


def plot_metric(
    *,
    dataframe: pd.DataFrame,
    y_column: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(7.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(
        dataframe["checkpoint_epoch"],
        dataframe[y_column],
        marker="o",
    )
    axis.set_xlabel("Эпоха checkpoint")
    axis.set_ylabel(y_label)
    axis.set_xticks(dataframe["checkpoint_epoch"].tolist())
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"График сохранён: {output_path}")


def create_plots(results: Sequence[dict], output_dir: Path) -> tuple[Path, Path]:
    dataframe = pd.DataFrame(results).sort_values("checkpoint_epoch")
    loss_path = output_dir / "evaluation_loss_available_checkpoints.png"
    eer_path = output_dir / "evaluation_eer_available_checkpoints.png"

    plot_metric(
        dataframe=dataframe,
        y_column="eval_loss",
        y_label="Функция потерь",
        output_path=loss_path,
    )
    plot_metric(
        dataframe=dataframe,
        y_column="eval_eer_percent",
        y_label="EER, %",
        output_path=eer_path,
    )
    return loss_path, eer_path


def log_to_wandb(
    *,
    results: Sequence[dict],
    loss_plot_path: Path,
    eer_plot_path: Path,
    entity: str | None,
    project: str,
    run_name: str,
    mode: str,
) -> str | None:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B не установлен. Выполните: pip install wandb"
        ) from error

    rows = sorted(results, key=lambda row: row["checkpoint_epoch"])
    config = {
        "evaluation_type": "post_hoc_available_checkpoints",
        "checkpoint_epochs": [row["checkpoint_epoch"] for row in rows],
        "checkpoint_paths": [row["checkpoint_path"] for row in rows],
        "selection_note": (
            "Evaluation metrics were not used for checkpoint selection."
        ),
    }

    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="evaluation",
        mode=mode,
        config=config,
    ) as run:
        run.define_metric("checkpoint_epoch")
        run.define_metric(
            "eval/loss",
            step_metric="checkpoint_epoch",
        )
        run.define_metric(
            "eval/eer_percent",
            step_metric="checkpoint_epoch",
        )

        for row in rows:
            run.log(
                {
                    "checkpoint_epoch": row["checkpoint_epoch"],
                    "eval/loss": row["eval_loss"],
                    "eval/eer_percent": row["eval_eer_percent"],
                }
            )

        table = wandb.Table(
            dataframe=pd.DataFrame(rows)[
                [
                    "checkpoint_epoch",
                    "eval_loss",
                    "eval_eer_percent",
                    "eer_threshold",
                    "number_of_objects",
                    "checkpoint_path",
                ]
            ]
        )
        run.log(
            {
                "eval/checkpoint_table": table,
                "eval/loss_plot": wandb.Image(str(loss_plot_path)),
                "eval/eer_plot": wandb.Image(str(eer_plot_path)),
            }
        )

        final_row = rows[-1]
        run.summary["final_checkpoint_epoch"] = final_row["checkpoint_epoch"]
        run.summary["final_eval_eer_percent"] = final_row[
            "eval_eer_percent"
        ]
        run.summary["final_eval_loss"] = final_row["eval_loss"]

        run_url = run.url

    print(f"W&B run: {run_url}")
    return run_url


def add_common_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help=(
            "Каталог для рекурсивного поиска checkpoint-файлов. "
            "Можно указать несколько раз."
        ),
    )


def default_search_roots(project_root: Path) -> list[str]:
    candidates = [
        project_root / "saved",
        Path("/kaggle/working/final_run"),
        Path("/kaggle/working") / project_root.name / "saved",
    ]
    return [str(path) for path in candidates if path.exists()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Поиск checkpoint-файлов и post-hoc оценка LCNN "
            "на ASVspoof 2019 LA evaluation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="Найти checkpoint-файлы и показать их реальные эпохи.",
    )
    add_common_search_arguments(list_parser)
    list_parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoint_inventory.csv"),
        help="Путь для CSV-инвентаризации.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Оценить checkpoint-файлы, построить графики и записать W&B run.",
    )
    add_common_search_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--epoch",
        type=int,
        action="append",
        default=[],
        help=(
            "Эпоха для автоматического выбора checkpoint. "
            "Можно указать несколько раз."
        ),
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Явный путь к checkpoint. Можно указать несколько раз. "
            "Имеет приоритет над --epoch."
        ),
    )
    evaluate_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/ASVspoof2019_LA"),
        help="Корень LA-части ASVspoof 2019.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoint_evaluation"),
        help="Каталог для CSV и PNG.",
    )
    evaluate_parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda или cuda:N.",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--num-workers", type=int, default=2)
    evaluate_parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity/team. Если не указан, используется настройка аккаунта.",
    )
    evaluate_parser.add_argument(
        "--wandb-project",
        default="voice-anti-spoofing-lcnn",
    )
    evaluate_parser.add_argument(
        "--wandb-run-name",
        default="lcnn-evaluation-checkpoints",
    )
    evaluate_parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    evaluate_parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Не отправлять результаты в W&B.",
    )

    return parser


def command_list(args, project_root: Path) -> int:
    raw_roots = args.search_root or default_search_roots(project_root)
    roots = normalize_roots(raw_roots)
    if not roots:
        raise FileNotFoundError(
            "Нет существующих каталогов поиска. Укажите --search-root."
        )

    print("Каталоги поиска:")
    for root in roots:
        print(f"  {root}")

    inventory = discover_checkpoints(roots)
    print_inventory(inventory)
    save_inventory(inventory, args.output.resolve())
    return 0


def command_evaluate(args, project_root: Path) -> int:
    os.chdir(project_root)

    if args.checkpoint:
        checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
        missing = [path for path in checkpoints if not path.exists()]
        if missing:
            missing_text = "\n".join(f"  {path}" for path in missing)
            raise FileNotFoundError(
                f"Не найдены checkpoint-файлы:\n{missing_text}"
            )
    else:
        if not args.epoch:
            raise ValueError(
                "Укажите --checkpoint или нужные эпохи через --epoch."
            )

        raw_roots = args.search_root or default_search_roots(project_root)
        roots = normalize_roots(raw_roots)
        if not roots:
            raise FileNotFoundError(
                "Нет существующих каталогов поиска. Укажите --search-root."
            )
        inventory = discover_checkpoints(roots)
        print_inventory(inventory)
        checkpoints = choose_checkpoints_by_epoch(inventory, args.epoch)

    epochs_and_paths = [
        (load_checkpoint_epoch(path), path) for path in checkpoints
    ]
    epochs = [epoch for epoch, _ in epochs_and_paths]
    if len(epochs) != len(set(epochs)):
        raise ValueError(
            "Переданы несколько checkpoint-файлов одной эпохи."
        )

    epochs_and_paths.sort(key=lambda item: item[0])
    checkpoints = [path for _, path in epochs_and_paths]

    print("Выбранные checkpoint-файлы:")
    for epoch, path in epochs_and_paths:
        print(f"  epoch {epoch}: {path}")

    dataset_root = args.dataset_root.expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (project_root / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}")

    config = build_evaluation_config(
        project_root=project_root,
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    from src.datasets.data_utils import get_dataloaders
    from src.loss import AntiSpoofingLoss
    from src.utils.init_utils import set_random_seed

    set_random_seed(1)
    dataloaders, batch_transforms = get_dataloaders(config, device)
    eval_loader = dataloaders["eval"]
    inference_transforms = batch_transforms.get("inference")

    try:
        from hydra.utils import instantiate
    except ImportError as error:
        raise RuntimeError(
            "Hydra не установлена. Выполните: pip install hydra-core"
        ) from error

    model = instantiate(config.model).to(device)
    criterion = AntiSpoofingLoss().to(device)

    results: list[dict] = []
    for checkpoint_path in checkpoints:
        results.append(
            evaluate_one_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                criterion=criterion,
                eval_loader=eval_loader,
                inference_transforms=inference_transforms,
                device=device,
            )
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    results.sort(key=lambda row: row["checkpoint_epoch"])

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "eval_checkpoints.csv"
    save_results_csv(results, results_csv)
    loss_plot_path, eer_plot_path = create_plots(results, output_dir)

    if not args.skip_wandb:
        log_to_wandb(
            results=results,
            loss_plot_path=loss_plot_path,
            eer_plot_path=eer_plot_path,
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            mode=args.wandb_mode,
        )

    print("\nГотово.")
    print(f"CSV: {results_csv}")
    print(f"Eval loss plot: {loss_plot_path}")
    print(f"Eval EER plot: {eer_plot_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = repository_root()

    try:
        if args.command == "list":
            return command_list(args, project_root)
        if args.command == "evaluate":
            return command_evaluate(args, project_root)
        parser.error(f"Неизвестная команда: {args.command}")
    except Exception as error:  # noqa: BLE001 - clean CLI error
        print(f"\nОШИБКА: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
