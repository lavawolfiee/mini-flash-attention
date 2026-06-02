import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "runtime": ("median_ms", "Runtime, median ms", "runtime_bars.png"),
    "memory": ("peak_memory_mb", "Peak allocated memory, MB", "memory_bars.png"),
    "tflops": ("tflops_s", "TFLOPs/s", "tflops_bars.png"),
}

BACKEND_ORDER = ["fa2", "fa1", "ours", "wmma", "torch"]

BACKEND_LABELS = {
    "fa2": "FlashAttention-2",
    "fa1": "FlashAttention-1",
    "ours": "Ours (CuTe)",
    "wmma": "Ours (WMMA)",
    "torch": "Pytorch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark_forward.py JSONL results.")
    parser.add_argument("inputs", type=Path, nargs="+", help="JSONL result files.")
    parser.add_argument("--out-dir", type=Path, default=Path("benchs/results"))
    parser.add_argument("--title", default="Forward attention benchmark")
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.setdefault("median_ms", row["mean_ms"])
                    row.setdefault("tflops_s", attention_tflops_s(row))
                    rows.append(row)
    return rows


def attention_tflops_s(row: dict) -> float:
    runtime_ms = row.get("median_ms", row["mean_ms"])
    if runtime_ms <= 0:
        return 0.0

    seq_len = row["seq_len"]
    if row["causal"]:
        attended_tokens = seq_len * (seq_len + 1) / 2
    else:
        attended_tokens = seq_len * seq_len

    flops = 4 * row["batch"] * row["heads"] * attended_tokens * row["head_dim"]
    return flops / (runtime_ms / 1000) / 1e12


def latest_rows(rows: list[dict]) -> list[dict]:
    by_case = {}
    for row in rows:
        key = (
            row["backend"],
            row["batch"],
            row["heads"],
            row["seq_len"],
            row["head_dim"],
            row["causal"],
            row["dtype"],
        )
        by_case[key] = row
    return list(by_case.values())


def grouped_values(rows: list[dict], metric: str) -> tuple[list[int], list[str], dict[tuple[int, str], float]]:
    seq_lens = sorted({row["seq_len"] for row in rows})
    available_backends = {row["backend"] for row in rows}
    ordered_backends = [backend for backend in BACKEND_ORDER if backend in available_backends]
    extra_backends = sorted(available_backends - set(BACKEND_ORDER))
    backends = ordered_backends + extra_backends
    values = {(row["seq_len"], row["backend"]): row[metric] for row in rows}
    return seq_lens, backends, values


def plot_metric(rows: list[dict], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    seq_lens, backends, values = grouped_values(rows, metric)
    x_positions = list(range(len(seq_lens)))
    total_width = 0.82
    bar_width = total_width / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    fig.suptitle(title)

    for backend_index, backend in enumerate(backends):
        offset = (backend_index - (len(backends) - 1) / 2) * bar_width
        points = [
            (x, values[(seq_len, backend)])
            for x, seq_len in zip(x_positions, seq_lens)
            if (seq_len, backend) in values
        ]
        if not points:
            continue

        positions = [x + offset for x, _ in points]
        heights = [height for _, height in points]
        bars = ax.bar(positions, heights, width=bar_width, label=BACKEND_LABELS.get(backend, backend))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(seq_len) for seq_len in seq_lens])
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncols=min(4, len(backends)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


def describe_run(rows: list[dict]) -> str:
    first = rows[0]
    return (
        f"B={first['batch']}, H={first['heads']}, D={first['head_dim']}, "
        f"dtype={first['dtype']}, causal={first['causal']}"
    )


def main() -> None:
    args = parse_args()
    rows = latest_rows(load_rows(args.inputs))
    if not rows:
        raise RuntimeError("No benchmark rows found")

    groups = defaultdict(list)
    for row in rows:
        key = (row["batch"], row["heads"], row["head_dim"], row["causal"], row["dtype"])
        groups[key].append(row)

    if len(groups) != 1:
        raise RuntimeError("Plot one benchmark shape configuration at a time")

    rows = next(iter(groups.values()))
    title = f"{args.title}\n{describe_run(rows)}"

    for _, (metric, ylabel, filename) in METRICS.items():
        plot_metric(rows, metric, ylabel, title, args.out_dir / filename)


if __name__ == "__main__":
    main()
