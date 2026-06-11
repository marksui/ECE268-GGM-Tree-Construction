from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt


PRFS = ["Spongent", "Keccak"]
PLATFORMS = ["CPU", "GPU"]

COLORS = {
    "CPU": "#182B49",
    "GPU": "#FFCD00",
    "Keccak": "#00629B",
    "Spongent": "#C69214",
}


def find_repo(start):
    path = Path(start).resolve()
    while path != path.parent:
        if (path / "bench" / "plot_results.py").exists():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find repo folder")


def load_results(path):
    rows = []
    header = None

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("prf,platform,depth"):
            header = next(csv.reader([line]))
            continue

        if line.startswith("Spongent,") or line.startswith("Keccak,"):
            if header is None:
                header = ["prf", "platform", "depth", "leaves", "best_ms", "leaves_per_sec"]

            values = next(csv.reader([line]))
            item = dict(zip(header, values))

            if item.get("best_ms") == "FAIL":
                continue

            rows.append({
                "prf": item["prf"],
                "platform": item["platform"],
                "depth": int(item["depth"]),
                "leaves": int(item["leaves"]),
                "best_ms": float(item["best_ms"]),
                "leaves_per_sec": float(item["leaves_per_sec"]),
            })

    if len(rows) == 0:
        raise ValueError("No benchmark rows found")

    return rows


def get_value(rows, prf, platform, depth, column):
    for row in rows:
        if row["prf"] == prf and row["platform"] == platform and row["depth"] == depth:
            return row[column]
    raise ValueError(f"Missing {prf} {platform} depth {depth}")


def available_depths(rows):
    return sorted(set(row["depth"] for row in rows))


def complete_depths(rows):
    depths = []
    for depth in available_depths(rows):
        ok = True
        for prf in PRFS:
            for platform in PLATFORMS:
                try:
                    get_value(rows, prf, platform, depth, "best_ms")
                except ValueError:
                    ok = False
        if ok:
            depths.append(depth)
    return depths


def finish_chart(ax):
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(fig, figure_dir, name):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_speedup_vs_depth(rows, figure_dir):
    depths = complete_depths(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

    for prf in PRFS:
        speedups = []
        for depth in depths:
            cpu_rate = get_value(rows, prf, "CPU", depth, "leaves_per_sec")
            gpu_rate = get_value(rows, prf, "GPU", depth, "leaves_per_sec")
            speedups.append(gpu_rate / cpu_rate)

        ax.plot(
            depths,
            speedups,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=COLORS[prf],
            label=prf,
        )

        first_gpu_win = None
        for depth, speedup in zip(depths, speedups):
            if speedup > 1.0:
                first_gpu_win = (depth, speedup)
                break

        if first_gpu_win is not None:
            depth, speedup = first_gpu_win
            ax.scatter([depth], [speedup], s=120, marker="*", color=COLORS[prf], edgecolor="black", zorder=3)
            ax.annotate(
                f"{prf}: GPU first wins at depth {depth}",
                xy=(depth, speedup),
                xytext=(8, 14),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8},
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1x = CPU and GPU equal")
    ax.set_title("GPU Speedup vs Tree Depth")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("GPU speedup over CPU")
    ax.set_xticks(depths)
    ax.legend()
    finish_chart(ax)
    return save_plot(fig, figure_dir, "speedup_vs_depth.png")


def choose_throughput_depth(rows):
    depths = complete_depths(rows)
    if 20 in depths:
        return 20
    if 12 in depths:
        return 12
    return depths[-1]


def plot_throughput_comparison(rows, figure_dir, depth=None):
    if depth is None:
        depth = choose_throughput_depth(rows)

    labels = ["CPU-Spongent", "GPU-Spongent", "CPU-Keccak", "GPU-Keccak"]
    values = [
        get_value(rows, "Spongent", "CPU", depth, "leaves_per_sec"),
        get_value(rows, "Spongent", "GPU", depth, "leaves_per_sec"),
        get_value(rows, "Keccak", "CPU", depth, "leaves_per_sec"),
        get_value(rows, "Keccak", "GPU", depth, "leaves_per_sec"),
    ]
    bar_colors = [COLORS["CPU"], COLORS["GPU"], COLORS["CPU"], COLORS["GPU"]]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_title(f"Throughput Comparison at Depth {depth}")
    ax.set_xlabel("PRF and platform")
    ax.set_ylabel("Leaves generated per second (log scale)")
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    finish_chart(ax)
    return save_plot(fig, figure_dir, f"throughput_comparison_depth_{depth}.png")


def plot_runtime_vs_depth(rows, figure_dir):
    depths = complete_depths(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

    series = [
        ("CPU-Spongent", "Spongent", "CPU", COLORS["CPU"], "-"),
        ("GPU-Spongent", "Spongent", "GPU", COLORS["GPU"], "-"),
        ("CPU-Keccak", "Keccak", "CPU", COLORS["CPU"], "--"),
        ("GPU-Keccak", "Keccak", "GPU", COLORS["GPU"], "--"),
    ]

    for label, prf, platform, color, style in series:
        values = [get_value(rows, prf, platform, depth, "best_ms") for depth in depths]
        ax.plot(depths, values, marker="o", linewidth=2.2, linestyle=style, color=color, label=label)

    ax.set_yscale("log")
    ax.set_title("Runtime vs Tree Depth")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Runtime in milliseconds (log scale)")
    ax.set_xticks(depths)
    ax.legend()
    finish_chart(ax)
    return save_plot(fig, figure_dir, "runtime_vs_depth.png")


def main():
    repo = find_repo(Path.cwd())
    docs_output_file = repo / "docs" / "benchmark_ggm_output.txt"
    output_file = repo / "output" / "benchmark_ggm_output.txt"
    full_depth_file = repo / "output" / "benchmark_ggm_8_12_16_20.txt"
    input_file = repo / "output" / "benchmark_ggm.txt"
    if output_file.exists() and output_file.stat().st_size > 0:
        input_file = output_file
    elif docs_output_file.exists() and docs_output_file.stat().st_size > 0:
        input_file = docs_output_file
    elif full_depth_file.exists() and full_depth_file.stat().st_size > 0:
        input_file = full_depth_file
    figure_dir = repo / "docs" / "results" / "figures"

    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        figure_dir = Path(sys.argv[2])

    rows = load_results(input_file)
    paths = [
        plot_speedup_vs_depth(rows, figure_dir),
        plot_throughput_comparison(rows, figure_dir),
        plot_runtime_vs_depth(rows, figure_dir),
    ]

    print("Saved figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
