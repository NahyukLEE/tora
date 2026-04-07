from collections import defaultdict
from itertools import chain
import re
from typing import Dict, List, Any

import lightning as L
import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table


def log_metrics_on_step(
    module: L.LightningModule,
    metrics: Dict[str, float],
    prefix: str = "train",
):
    """Log per-step scalars only on rank 0."""
    for name, value in metrics.items():
        module.log(
            f"{prefix}/{name}",
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            rank_zero_only=True,
        )


def log_metrics_on_epoch(
    module: L.LightningModule,
    metrics: Dict[str, float],
    prefix: str = "val",
):
    """Log per-epoch scalars, automatically synced & averaged across ranks."""
    for name, value in metrics.items():
        module.log(
            f"{prefix}/{name}",
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            rank_zero_only=True,
        )

def print_eval_table(
    results: list[dict[str, float]],
    dataset_names: list[str],
    dataset_counts: list[int] | None = None,
    digits: int = 4,
) -> None:
    """
    Pretty-print evaluation results with Rich, split into Avg and BoN sections.

    Supports two key formats:
      - Meter format: "test/{dataset}/{metric}" (from MetricsMeter + log_metrics_on_epoch)
      - Legacy format: "{metric}/dataloader_idx_{idx}" (from self.log_dict in test_step)

    Args:
        results: List of dicts from trainer.test().
        dataset_names: Dataset names corresponding to dataloader indices.
        dataset_counts: Number of samples per dataset for weighted averaging (legacy only).
        digits: Number of decimal places for floats (minimum 4).
    """
    digits = max(digits, 4)
    fmt = f"{{:.{digits}f}}"

    def _to_float(val) -> float | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, "item"):  # torch.Tensor
            return val.item()
        return None

    # Merge all result dicts into one
    merged: Dict[str, Any] = {}
    for d in results:
        merged.update(d)

    # Detect format: meter keys start with "test/{dataset_name}/"
    meter_pattern = re.compile(r"^test/(.+?)/(.+)$")
    has_meter_keys = any(meter_pattern.match(k) for k in merged)

    if has_meter_keys:
        # Meter format: test/{dataset}/{metric}
        per_ds: Dict[str, Dict[str, float]] = {}
        for k, v in merged.items():
            m = meter_pattern.match(k)
            if not m:
                continue
            ds, metric = m.group(1), m.group(2)
            per_ds.setdefault(ds, {})[metric] = _to_float(v)

        # Separate "overall" from dataset columns
        overall = per_ds.pop("overall", {})
        ds_order = [n for n in dataset_names if n in per_ds]
        # Include any datasets the meter found that aren't in dataset_names
        for ds in sorted(per_ds):
            if ds not in ds_order:
                ds_order.append(ds)

        all_metrics = set()
        for ds_metrics in per_ds.values():
            all_metrics.update(ds_metrics.keys())
        avg_metrics = sorted(m for m in all_metrics if m.startswith("avg/"))
        bon_metrics = sorted(m for m in all_metrics if m.startswith("best_of_n/"))

        has_multiple = len(ds_order) > 1

        def _build_rows_meter(metric_list):
            rows = []
            for metric in metric_list:
                row = [metric]
                for ds in ds_order:
                    v = per_ds.get(ds, {}).get(metric)
                    row.append(fmt.format(v) if v is not None else "-")
                if has_multiple:
                    v = overall.get(metric)
                    row.append(fmt.format(v) if v is not None else "-")
                rows.append(row)
            return rows

        table = Table()
        table.add_column("Metrics", style="bold magenta", justify="left", no_wrap=True)
        for ds in ds_order:
            table.add_column(ds, style="cyan")
        if has_multiple:
            table.add_column("overall", style="bold green")

        for row in _build_rows_meter(avg_metrics):
            table.add_row(*row)
        if bon_metrics:
            table.add_section()
            for row in _build_rows_meter(bon_metrics):
                table.add_row(*row)

        Console().print(table)
        return

    # Legacy format: {metric}/dataloader_idx_{idx}
    idx_pattern = re.compile(r"dataloader_idx_(\d+)")
    per_idx: Dict[int, Dict[str, Any]] = {}
    single_dataloader = len(results) == 1 and not idx_pattern.search(next(iter(results[0])))
    for i, d in enumerate(results):
        if single_dataloader:
            idx = 0
        else:
            idx = int(idx_pattern.search(next(iter(d))).group(1))
        per_idx[idx] = d
    if single_dataloader:
        metric_pattern = re.compile(r"^(.+?/.+?)$")
    else:
        metric_pattern = re.compile(r"^(.+?/.+?)/dataloader_idx_\d+$")
    metrics = set()
    for d in results:
        for k in d:
            m = metric_pattern.match(k)
            if m:
                metrics.add(m.group(1))

    sorted_idxs = sorted(per_idx)
    has_multiple = len(sorted_idxs) > 1
    if dataset_counts is not None:
        weights = {idx: dataset_counts[idx] for idx in sorted_idxs}
    else:
        weights = None

    def _get_key(metric, idx):
        return metric if single_dataloader else f"{metric}/dataloader_idx_{idx}"

    def _weighted_avg(vals: list[tuple[int, float | None]]) -> str:
        valid = [(idx, v) for idx, v in vals if v is not None]
        if not valid:
            return "-"
        if weights is not None:
            w_sum = sum(weights[idx] * v for idx, v in valid)
            w_total = sum(weights[idx] for idx, _ in valid)
            return fmt.format(w_sum / w_total) if w_total > 0 else "-"
        else:
            return fmt.format(sum(v for _, v in valid) / len(valid))

    def _build_rows(metric_list):
        rows = []
        for metric in metric_list:
            row = [metric]
            vals = []
            for idx in sorted_idxs:
                v = _to_float(per_idx[idx].get(_get_key(metric, idx)))
                row.append(fmt.format(v) if v is not None else "-")
                vals.append((idx, v))
            if has_multiple:
                row.append(_weighted_avg(vals))
            rows.append(row)
        return rows

    avg_metrics = sorted(m for m in metrics if m.lower().startswith("avg/"))
    bon_metrics = sorted(m for m in metrics if m.lower().startswith("best_of_n/"))

    table = Table()
    table.add_column("Metrics", style="bold magenta", justify="left", no_wrap=True)
    for idx in sorted_idxs:
        table.add_column(dataset_names[idx], style="cyan")
    if has_multiple:
        table.add_column("weighted avg", style="bold green")

    for row in _build_rows(avg_metrics):
        table.add_row(*row)
    if bon_metrics:
        table.add_section()
        for row in _build_rows(bon_metrics):
            table.add_row(*row)

    Console().print(table)


class MetricsMeter:
    """Helper class for accumulating metrics for each dataset.

    Example:
        >>> metrics_meter = MetricsMeter(module)
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "A"],
                loss=torch.tensor([0.1, 0.2, 0.3]),
                acc=torch.tensor([0.9, 0.8, 0.7]),
            )
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "C"],
                loss=torch.tensor([0.4, 0.5, 0.6]),
                acc=torch.tensor([0.6, 0.5, 0.4]),
            )
        >>> results = metrics_meter.log_on_epoch_end()
        >>> print(results)
        {
            "A/loss": 0.2667,
            "A/acc": 0.7333,
            "B/loss": 0.35,
            "B/acc": 0.65,
            "C/loss": 0.6,
            "C/acc": 0.4,
            "overall/loss": 0.35,
            "overall/acc": 0.65,
        }
    """

    def __init__(self, module: L.LightningModule):
        self.module = module
        self.reset()

    def reset(self):
        self._sums = defaultdict(lambda: defaultdict(float))
        self._counts = defaultdict(lambda: defaultdict(int))
        self._metrics_seen = set()

    def add_metrics(self, dataset_names: List[str], **metrics: torch.Tensor):
        """Accumulate a batch of per-sample metrics."""
        if not metrics:
            return

        if any(ds == "overall" for ds in dataset_names):
            raise ValueError("'overall' is a reserved dataset name and cannot be used.")

        B = next(iter(metrics.values())).shape[0]
        if len(dataset_names) != B:
            raise ValueError(f"len(dataset_names)={len(dataset_names)} != batch size {B}")
        for k, t in metrics.items():
            if t.shape[0] != B:
                raise ValueError(f"metric '{k}' has shape {t.shape} != ({B},)")
            self._metrics_seen.add(k)

        for i, ds in enumerate(dataset_names):
            for k, t in metrics.items():
                v = t[i].item()
                self._sums[k][ds] += v
                self._counts[k][ds] += 1
                self._sums[k]["_overall"] += v
                self._counts[k]["_overall"] += 1

    def compute_average(self) -> Dict[str, torch.Tensor]:
        """Gather per-dataset sums/counts, and compute global averages."""
        # local dataset list
        local_ds = sorted(
            set(chain.from_iterable(self._counts[k].keys() for k in self._metrics_seen))
            - {"_overall"}
        )

        # gather global dataset list
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_ds)
            global_ds = sorted(set(chain.from_iterable(gathered)))
        else:
            global_ds = local_ds

        # flatten sums and counts with fixed order
        metrics = sorted(self._metrics_seen)
        N = len(global_ds) + 1
        flat_sums = []
        flat_counts = []
        for k in metrics:
            for ds in global_ds:
                flat_sums.append(self._sums[k].get(ds, 0.0))
                flat_counts.append(self._counts[k].get(ds, 0))
            flat_sums.append(self._sums[k].get("_overall", 0.0))
            flat_counts.append(self._counts[k].get("_overall", 0))

        device = getattr(self.module, "device", torch.device("cpu")) or torch.device("cpu")
        sums_t = torch.tensor(flat_sums, dtype=torch.float64, device=device)
        counts_t = torch.tensor(flat_counts, dtype=torch.float64, device=device)

        # all-reduce
        if world_size > 1:
            dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)

        # compute average metrics
        results: Dict[str, torch.Tensor] = {}
        for idx, k in enumerate(metrics):
            for j, ds in enumerate(global_ds + ["overall"]):
                pos = idx * N + j
                total_sum = sums_t[pos]
                total_count = counts_t[pos]
                avg = (
                    total_sum / total_count if total_count > 0
                    else torch.tensor(float("nan"), device=device)
                )
                results[f"{ds}/{k}"] = avg

        # reset for next epoch
        self.reset()
        return results
