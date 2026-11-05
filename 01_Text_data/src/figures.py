"""Regenerate the four-setting accuracy figure from exported metrics."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(csv_path, output_path):
    records = pd.read_csv(csv_path)
    x = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = [("val_accuracy", "Validation"),
               ("test_accuracy", "Test"),
               ("nonoverlap_test_accuracy", "Non-overlap test")]
    for offset, (key, label) in zip([-0.25, 0, 0.25], metrics):
        bars = ax.bar(x + offset, records[key] * 100, 0.25, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.tokenizer} {row.vocab_size:,}"
                        for row in records.itertuples()])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tokenizer Comparison - Four Settings")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
