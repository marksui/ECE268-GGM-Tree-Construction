import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

os.makedirs("plots", exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG         = "#FFFFFF"
PANEL      = "#F5F7FA"
BLUE       = "#1A7BBF"   # Spongent GPU
BLUE_DARK  = "#0F4C75"   # Spongent CPU
GOLD       = "#D4900A"   # Keccak GPU
GOLD_MUTED = "#B07A10"   # Keccak CPU
TEXT       = "#1A1A2E"
GRID       = "#D0D7E3"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi":       150,
})

# ── Raw data ──────────────────────────────────────────────────────────────────

s0 = {
    "depth":   [8,       12,        16,         20],
    "spu_cpu": [727.317, 11859.152, 179874.245, 2804658.541],
    "spu_gpu": [59.028,  108.581,   264.376,    1674.492],
    "kec_cpu": [1.311,   20.581,    320.665,    5346.094],
    "kec_gpu": [0.383,   0.492,     0.684,      5.189],
}

s1 = {
    "depth_both": [4,      6,       10,       14],
    "spu_cpu_m":  [39.759, 165.677, 2717.463, 43709.464],
    "spu_cpu_s":  [0.716,  0.428,   10.564,   162.022],
    "spu_gpu_m":  [13.907, 29.441,  63.114,   128.622],
    "spu_gpu_s":  [0.045,  0.028,   0.240,    18.395],
    "kec_cpu_m":  [0.079,  0.311,   4.970,    81.235],
    "kec_cpu_s":  [0.008,  0.029,   0.056,    0.825],
    "kec_gpu_m":  [0.174,  0.194,   0.258,    0.390],
    "kec_gpu_s":  [0.021,  0.014,   0.018,    0.037],
    "d18_spu_gpu_m": 488.450, "d18_spu_gpu_s": 1.923,
    "d18_kec_cpu_m": 1295.269,"d18_kec_cpu_s": 10.890,
    "d18_kec_gpu_m": 1.969,   "d18_kec_gpu_s": 0.090,
}

s2 = {
    "tpb":     [32,      64,      128,     256,     512],
    "mean_ms": [188.459, 175.577, 175.870, 194.095, 250.334],
    "std_ms":  [18.892,  0.075,   0.043,   0.265,   0.271],
    "lps":     [347747.38, 373261.04, 372638.63, 337648.63, 261794.11],
}

s3 = {
    "label":    ["Sp d12","Ke d12","Sp d16","Ke d16","Sp d20","Ke d20"],
    "build_ms": [83.972,  0.319,   194.074, 0.612,   1590.903, 4.970],
    "d2h_ms":   [0.038,   0.053,   0.291,   0.488,   6.574,    12.601],
}

# Mapping exact ground-truth throughput to bypass float precision issues
raw_lps_map = {
    ("Spongent", "CPU", 4): 402.42,   ("Spongent", "GPU", 4): 1150.47,
    ("Keccak", "CPU", 4): 202271.60,  ("Keccak", "GPU", 4): 91838.57,
    ("Spongent", "CPU", 6): 386.29,   ("Spongent", "GPU", 6): 2173.85,
    ("Keccak", "CPU", 6): 205506.43,  ("Keccak", "GPU", 6): 330239.36,
    ("Spongent", "CPU", 8): 351.98,   ("Spongent", "GPU", 8): 4336.90,
    ("Keccak", "CPU", 8): 195338.30,  ("Keccak", "GPU", 8): 669161.46,
    ("Spongent", "CPU", 10): 376.82,  ("Spongent", "GPU", 10): 16224.61,
    ("Keccak", "CPU", 10): 206053.63, ("Keccak", "GPU", 10): 3967370.41,
    ("Spongent", "CPU", 12): 345.39,  ("Spongent", "GPU", 12): 37723.06,
    ("Keccak", "CPU", 12): 199022.71, ("Keccak", "GPU", 12): 8326161.79,
    ("Spongent", "CPU", 14): 374.84,  ("Spongent", "GPU", 14): 127380.89,
    ("Keccak", "CPU", 14): 201686.32, ("Keccak", "GPU", 14): 41985025.03,
    ("Spongent", "CPU", 16): 364.34,  ("Spongent", "GPU", 16): 247889.16,
    ("Keccak", "CPU", 16): 204375.26, ("Keccak", "GPU", 16): 95835578.72,
    ("Spongent", "GPU", 18): 536685.06,("Keccak", "CPU", 18): 202385.72, ("Keccak", "GPU", 18): 133152507.94,
    ("Spongent", "CPU", 20): 373.87,  ("Spongent", "GPU", 20): 626205.28,
    ("Keccak", "CPU", 20): 196138.70, ("Keccak", "GPU", 20): 202068562.50
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def merge_sort(d_list, ms_list, std_list=None):
    std = std_list if std_list is not None else [0] * len(d_list)
    rows = sorted(zip(d_list, ms_list, std))
    d, m, s = zip(*rows)
    return list(d), list(m), list(s)

def get_true_lps(prf, platform, depth_list):
    return [raw_lps_map.get((prf, platform, d), 0.0) for d in depth_list]

def style_ax(ax, all_depths, ylabel, title):
    ax.set_yscale("log")
    ax.set_xlabel("Tree Depth", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title)
    ax.set_xticks(sorted(set(all_depths)))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

def add_labels(ax, x, y, fmt_func, color, y_offset_factor=1.15):
    """Safely plots explicit data labels on a log-scaled canvas without overlaps."""
    for xi, yi in zip(x, y):
        if yi == 0: continue
        ax.text(xi, yi * y_offset_factor, fmt_func(yi), 
                color=color, fontsize=8, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none", alpha=0.75))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — GPU Speedup vs Tree Depth
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup():
    spu_d, spu_c, _ = merge_sort(s0["depth"] + s1["depth_both"], s0["spu_cpu"] + s1["spu_cpu_m"])
    _,     spu_g, _ = merge_sort(s0["depth"] + s1["depth_both"], s0["spu_gpu"] + s1["spu_gpu_m"])
    spu_speedup = [c / g for c, g in zip(spu_c, spu_g)]

    kec_d, kec_c, _ = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_cpu"] + s1["kec_cpu_m"] + [s1["d18_kec_cpu_m"]])
    _,     kec_g, _ = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_gpu"] + s1["kec_gpu_m"] + [s1["d18_kec_gpu_m"]])
    kec_speedup = [c / g for c, g in zip(kec_c, kec_g)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(spu_d, spu_speedup, color=BLUE, marker="o", linewidth=2.2, markersize=7, label="Spongent GPU / CPU")
    ax.plot(kec_d, kec_speedup, color=GOLD, marker="s", linewidth=2.2, markersize=7, label="Keccak GPU / CPU")

    # Add labels
    add_labels(ax, spu_d, spu_speedup, lambda v: f"{v:,.1f}x", BLUE, 1.2)
    add_labels(ax, kec_d, kec_speedup, lambda v: f"{v:,.1f}x", GOLD, 0.75)

    ax.axhline(1.0, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.35)
    
    style_ax(ax, spu_d + kec_d, "GPU Speedup (× CPU runtime)", "Plot 1 — GPU Speedup over CPU vs Tree Depth")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}×"))
    ax.legend(framealpha=0.7, edgecolor=GRID, loc="upper left")

    fig.tight_layout()
    fig.savefig("plots/plot1_speedup.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot1_speedup.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Throughput vs Tree Depth
# ─────────────────────────────────────────────────────────────────────────────
def plot_throughput():
    sc_d, sc_ms, _ = merge_sort(s0["depth"] + s1["depth_both"], s0["spu_cpu"] + s1["spu_cpu_m"])
    sg_d, sg_ms, _ = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["spu_gpu"] + s1["spu_gpu_m"] + [s1["d18_spu_gpu_m"]])
    kc_d, kc_ms, _ = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_cpu"] + s1["kec_cpu_m"] + [s1["d18_kec_cpu_m"]])
    kg_d, kg_ms, _ = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_gpu"] + s1["kec_gpu_m"] + [s1["d18_kec_gpu_m"]])

    sc_lps = get_true_lps("Spongent", "CPU", sc_d)
    sg_lps = get_true_lps("Spongent", "GPU", sg_d)
    kc_lps = get_true_lps("Keccak", "CPU", kc_d)
    kg_lps = get_true_lps("Keccak", "GPU", kg_d)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(sc_d, sc_lps, color=BLUE_DARK,  marker="o", linestyle="-", linewidth=2.2, label="CPU Spongent")
    ax.plot(sg_d, sg_lps, color=BLUE,       marker="s", linestyle="--", linewidth=2.2, label="GPU Spongent")
    ax.plot(kc_d, kc_lps, color=GOLD_MUTED, marker="^", linestyle="-", linewidth=2.2, label="CPU Keccak")
    ax.plot(kg_d, kg_lps, color=GOLD,       marker="D", linestyle="--", linewidth=2.2, label="GPU Keccak")

    # Format helpers for tight label structures
    fmt = lambda v: f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.1f}k" if v >= 1e3 else f"{v:.0f}")
    add_labels(ax, sc_d, sc_lps, fmt, BLUE_DARK, 0.70)
    add_labels(ax, sg_d, sg_lps, fmt, BLUE, 1.35)
    add_labels(ax, kc_d, kc_lps, fmt, GOLD_MUTED, 0.75)
    add_labels(ax, kg_d, kg_lps, fmt, GOLD, 1.30)

    style_ax(ax, sc_d + sg_d + kc_d + kg_d, "Throughput (leaves / sec, log scale)", "Plot 2 — Throughput vs Tree Depth")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
    ax.legend(framealpha=0.7, edgecolor=GRID, loc="lower left")

    fig.tight_layout()
    fig.savefig("plots/plot2_throughput.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot2_throughput.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Runtime vs Depth
# ─────────────────────────────────────────────────────────────────────────────
def draw_line_with_labels(ax, d, m, s, color, ls, marker, label, offset, fmt_func):
    ax.plot(d, m, color=color, linestyle=ls, marker=marker, linewidth=2.2, markersize=6, label=label, zorder=3)
    if np.any(np.array(s) > 0):
        ax.fill_between(d, np.maximum(np.array(m) - np.array(s), 1e-6), np.array(m) + np.array(s), color=color, alpha=0.15, zorder=2)
    add_labels(ax, d, m, fmt_func, color, offset)

def plot_runtime():
    sc_d, sc_ms, sc_sd = merge_sort(s0["depth"] + s1["depth_both"], s0["spu_cpu"] + s1["spu_cpu_m"], [0]*4 + s1["spu_cpu_s"])
    sg_d, sg_ms, sg_sd = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["spu_gpu"] + s1["spu_gpu_m"] + [s1["d18_spu_gpu_m"]], [0]*4 + s1["spu_gpu_s"] + [s1["d18_spu_gpu_s"]])
    kc_d, kc_ms, kc_sd = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_cpu"] + s1["kec_cpu_m"] + [s1["d18_kec_cpu_m"]], [0]*4 + s1["kec_cpu_s"] + [s1["d18_kec_cpu_s"]])
    kg_d, kg_ms, kg_sd = merge_sort(s0["depth"] + s1["depth_both"] + [18], s0["kec_gpu"] + s1["kec_gpu_m"] + [s1["d18_kec_gpu_m"]], [0]*4 + s1["kec_gpu_s"] + [s1["d18_kec_gpu_s"]])

    fig, ax = plt.subplots(figsize=(11, 6.5))

    fmt = lambda v: f"{v/1e3:.1f}s" if v >= 1e3 else f"{v:.1f}ms"
    draw_line_with_labels(ax, sc_d, sc_ms, sc_sd, BLUE_DARK,  "-",  "o", "CPU Spongent", 1.3, fmt)
    draw_line_with_labels(ax, sg_d, sg_ms, sg_sd, BLUE,       "--", "s", "GPU Spongent", 0.75, fmt)
    draw_line_with_labels(ax, kc_d, kc_ms, kc_sd, GOLD_MUTED, "-",  "^", "CPU Keccak",   1.3, fmt)
    draw_line_with_labels(ax, kg_d, kg_ms, kg_sd, GOLD,       "--", "D", "GPU Keccak",   0.7, fmt)

    style_ax(ax, sc_d + sg_d + kc_d + kg_d, "Wall-clock Time (ms, log scale)", "Plot 3 — Runtime vs Tree Depth")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.1f}s" if v >= 1e3 else f"{v:,.0f} ms"))
    ax.legend(framealpha=0.7, edgecolor=GRID, loc="upper left")

    fig.tight_layout()
    fig.savefig("plots/plot3_runtime.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot3_runtime.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Block-size sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def plot_blocksize():
    tpb  = s2["tpb"]
    vals = s2["lps"]
    sd   = s2["std_ms"]
    mean = s2["mean_ms"]
    lps_std = [v * (s / m) for v, s, m in zip(vals, sd, mean)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(tpb, vals, color=BLUE, marker="o", linewidth=2.2, markersize=8, zorder=3, label="Throughput")
    ax.fill_between(tpb, [v - e for v, e in zip(vals, lps_std)], [v + e for v, e in zip(vals, lps_std)], color=BLUE, alpha=0.15, zorder=2)

    # Dynamic data labeling for throughput values
    for x, y in zip(tpb, vals):
        ax.text(x, y + 4000, f"{y/1e3:.1f}k l/s", color=TEXT, fontsize=8.5, ha="center")

    winner_idx = int(np.argmax(vals))
    ax.plot(tpb[winner_idx], vals[winner_idx], marker="*", color=GOLD, markersize=16, zorder=4, label=f"Optimal ({tpb[winner_idx]} tpb)")

    ax.set_xlabel("Threads per Block (tpb)", labelpad=8)
    ax.set_ylabel("Throughput (leaves / sec)", labelpad=8)
    ax.set_title("Plot 4 — Block-Size Sensitivity (Spongent GPU, depth = 16)")
    ax.set_xticks(tpb)
    ax.set_ylim(min(vals) - 30000, max(vals) + 30000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(framealpha=0.7, edgecolor=GRID)

    fig.tight_layout()
    fig.savefig("plots/plot4_blocksize.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot4_blocksize.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Memory Transfer Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_memory():
    sp_depths = [12, 16, 20]
    sp_build  = [s3["build_ms"][0], s3["build_ms"][2], s3["build_ms"][4]]
    sp_d2h    = [s3["d2h_ms"][0],   s3["d2h_ms"][2],   s3["d2h_ms"][4]]
    ke_depths = [12, 16, 20]
    ke_build  = [s3["build_ms"][1], s3["build_ms"][3], s3["build_ms"][5]]
    ke_d2h    = [s3["d2h_ms"][1],   s3["d2h_ms"][3],   s3["d2h_ms"][5]]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(sp_depths, sp_build, color=BLUE,      marker="o", linestyle="-",  linewidth=2.2, label="Spongent — GPU Compute")
    ax.plot(sp_depths, sp_d2h,   color=BLUE_DARK, marker="o", linestyle="--", linewidth=2.2, label="Spongent — D2H Transfer")
    ax.plot(ke_depths, ke_build, color=GOLD,       marker="s", linestyle="-",  linewidth=2.2, label="Keccak — GPU Compute")
    ax.plot(ke_depths, ke_d2h,   color=GOLD_MUTED, marker="s", linestyle="--", linewidth=2.2, label="Keccak — D2H Transfer")

    # Explicit inline data labeling via custom scale offsets
    fmt = lambda v: f"{v:.3f}ms" if v < 1 else f"{v:,.1f}ms"
    add_labels(ax, sp_depths, sp_build, fmt, BLUE, 1.25)
    add_labels(ax, sp_depths, sp_d2h,   fmt, BLUE_DARK, 0.65)
    add_labels(ax, ke_depths, ke_build, fmt, GOLD, 1.25)
    add_labels(ax, ke_depths, ke_d2h,   fmt, GOLD_MUTED, 0.65)

    ax.set_yscale("log")
    ax.set_xlabel("Tree Depth", labelpad=8)
    ax.set_ylabel("Time for transfer(ms, log scale)", labelpad=8)
    ax.set_title("Plot 5 — Memory Transfer Breakdown (GPU compute vs D2H)")
    ax.set_xticks(sp_depths)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(framealpha=0.7, edgecolor=GRID, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig("plots/plot5_memory.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot5_memory.png")

def plot_crossover():
    # Filter for the overhead/crossover zone (Depths 4 to 10)
    crossover_depths = [4, 6, 8, 10]
    
    # Extract matching clean data pools from s0 and s1
    spu_cpu = [39.759, 165.677, 727.317, 2717.463]
    spu_gpu = [13.907, 29.441,  59.028,  63.114]
    
    kec_cpu = [0.079,   0.311,   1.311,   4.970]
    kec_gpu = [0.174,   0.194,   0.383,   0.258]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True)

    # ── PANEL 1: SPONGENT OVERVIEW ───────────────────────────────────────────
    ax1.plot(crossover_depths, spu_cpu, color=BLUE_DARK, marker="o", linewidth=2.5, label="CPU Spongent")
    ax1.plot(crossover_depths, spu_gpu, color=BLUE, marker="s", linestyle="--", linewidth=2.5, label="GPU Spongent")
    
    # Labels for Spongent
    for x, c_y, g_y in zip(crossover_depths, spu_cpu, spu_gpu):
        ax1.text(x, c_y * 1.2, f"{c_y:.1f}ms", color=BLUE_DARK, fontsize=8, ha="center")
        ax1.text(x, g_y * 0.7, f"{g_y:.1f}ms", color=BLUE, fontsize=8, ha="center")

    ax1.axvline(x=4.0, color="#E63946", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.text(4.1, 1500, "Crossover < Depth 4\n(GPU wins instantly)", color="#E63946", fontsize=9, fontweight="bold")
    
    ax1.set_yscale("log")
    ax1.set_title("Spongent Crossover Zone")
    ax1.set_ylabel("Execution Time (ms, log scale)")
    ax1.set_xlabel("Tree Depth")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── PANEL 2: KECCAK OVERVIEW ─────────────────────────────────────────────
    ax2.plot(crossover_depths, kec_cpu, color=GOLD_MUTED, marker="^", linewidth=2.5, label="CPU Keccak")
    ax2.plot(crossover_depths, kec_gpu, color=GOLD, marker="D", linestyle="--", linewidth=2.5, label="GPU Keccak")
    
    # Labels for Keccak
    for x, c_y, g_y in zip(crossover_depths, kec_cpu, kec_gpu):
        ax2.text(x, c_y * 1.2, f"{c_y:.2f}ms", color=GOLD_MUTED, fontsize=8, ha="center")
        ax2.text(x, g_y * 0.7, f"{g_y:.2f}ms", color=GOLD, fontsize=8, ha="center")

    # Mark the exact intersection point (around depth 5)
    ax2.axvline(x=5.0, color="#E63946", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.plot(5.0, 0.185, marker="X", color="#E63946", markersize=10, zorder=5)
    ax2.text(5.2, 0.04, "Crossover ~ Depth 5\nLaunch Overhead Overtaken", color="#E63946", fontsize=9, fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_title("Keccak Crossover Zone")
    ax2.set_xlabel("Tree Depth")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("GGM Execution Crossover Analysis (GPU Overhead vs Compute Intensity)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig("plots/plot6_crossover.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ plot6_crossover.png")
    
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating GGM Tree benchmark plots → plots/")
    plot_speedup()
    plot_throughput()
    plot_runtime()
    plot_blocksize()
    plot_memory()
    plot_crossover()
    print("Done.")