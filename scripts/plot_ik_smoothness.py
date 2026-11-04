"""Plot only overlap distributions from already saved IK derivative NPZ files.

Reads summary.json and its derivative manifest; never reads training datasets
or recomputes derivatives. All overlap samples contribute to the histograms.
Works with derivative archive format versions 1 and 2.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np


VIEW = "matched_training_intervals"
VERSIONS = ("mink", "three_stage")
STREAMS = ("state", "action")
ARMS = ("left", "right")
ORDERS = {
    "velocity": ("velocity_deg_s", "velocity_frame_indices", "first_derivative", "deg/s", 1.0),
    "acceleration": ("acceleration_deg_s2", "acceleration_frame_indices", "second_derivative", "deg/s^2", 10.0),
}
COLORS = {"mink": "#3366AA", "three_stage": "#D97718"}
LABELS = {"mink": "Mink", "three_stage": "Three-stage"}


def read_json(path):
    return json.loads(path.read_text())


def bin_edges(maximum, linear_threshold):
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("Invalid maximum in input summary")
    cap = np.nextafter(max(maximum, linear_threshold), np.inf)
    center = np.linspace(0, linear_threshold, 25)
    tail = np.geomspace(linear_threshold, cap, 121)[1:] if cap > linear_threshold else []
    positive = np.unique(np.r_[center, tail])
    return np.r_[-positive[:0:-1], positive], positive


@dataclass
class Distribution:
    signed_edges: np.ndarray
    absolute_edges: np.ndarray

    def __post_init__(self):
        self.signed_counts = np.zeros((6, len(self.signed_edges) - 1), dtype=np.int64)
        self.absolute_counts = np.zeros((6, len(self.absolute_edges) - 1), dtype=np.int64)
        self.samples = 0
        self.sum_abs = np.zeros(6)
        self.sum_square = np.zeros(6)
        self.max_abs = np.zeros(6)
        self.zeros = np.zeros(6, dtype=np.int64)

    def add(self, values):
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError("Expected all finite signed derivative values with shape (N,6)")
        self.samples += len(values)
        if not len(values):
            return
        absolute = np.abs(values)
        self.sum_abs += absolute.sum(axis=0)
        self.sum_square += np.square(values).sum(axis=0)
        self.max_abs = np.maximum(self.max_abs, absolute.max(axis=0))
        self.zeros += (values == 0).sum(axis=0)
        for joint in range(6):
            signed = np.histogram(values[:, joint], self.signed_edges)[0]
            magnitude = np.histogram(absolute[:, joint], self.absolute_edges)[0]
            if signed.sum() != len(values) or magnitude.sum() != len(values):
                raise ValueError("Values outside histogram range; saved arrays differ from the input summary")
            self.signed_counts[joint] += signed
            self.absolute_counts[joint] += magnitude

    def report(self):
        if not self.samples:
            return {"samples_per_joint": 0}
        return {
            "samples_per_joint": self.samples, "scalar_samples": 6 * self.samples,
            "rms": float(np.sqrt(self.sum_square.mean() / self.samples)),
            "mean_absolute": float(self.sum_abs.mean() / self.samples),
            "maximum_absolute": float(self.max_abs.max()),
            "per_joint_rms": np.sqrt(self.sum_square / self.samples).tolist(),
            "per_joint_maximum_absolute": self.max_abs.tolist(),
            "zero_fraction": float(self.zeros.sum() / (6 * self.samples)),
        }

    def check_summary(self, expected):
        actual = self.report()
        if actual["samples_per_joint"] != expected["samples_per_joint"]:
            raise ValueError("Saved derivative sample count differs from summary")
        if self.samples:
            for key in ("rms", "mean_absolute", "maximum_absolute", "per_joint_rms"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-8,
                                           err_msg=f"Saved derivatives differ from summary: {key}")


def resolve_derivatives(input_dir, summary):
    recorded = Path(summary["derivatives"]["directory"])
    alternatives = (input_dir / "summary.derivatives", recorded)
    for root in alternatives:
        if (root / "manifest.json").is_file():
            return root.resolve()
    raise FileNotFoundError("No saved derivative manifest found for this completed result")


def read_distributions(input_dir):
    summary = read_json(input_dir / "summary.json")
    if summary.get("status") != "complete" or VIEW not in summary.get("views", {}):
        raise ValueError("Expected a completed result containing matched_training_intervals")
    raw_dir = resolve_derivatives(input_dir, summary)
    manifest = read_json(raw_dir / "manifest.json")
    files = manifest.get("files", [])
    if (manifest.get("status") != "complete" or len(files) != summary["sequences"]
            or len({row["file"] for row in files}) != len(files)):
        raise ValueError("Derivative manifest is incomplete or contains duplicate episodes")
    stats = summary["views"][VIEW]
    distributions = {}
    for order, (_, _, field_name, _, threshold) in ORDERS.items():
        for stream in STREAMS:
            for arm in ARMS:
                maximum = max(stats[version][stream][arm][field_name].get("maximum_absolute", 0)
                              for version in VERSIONS)
                signed, absolute = bin_edges(maximum, threshold)
                for version in VERSIONS:
                    distributions[order, stream, arm, version] = Distribution(signed, absolute)
    for number, row in enumerate(files, 1):
        path = (raw_dir / row["file"]).resolve()
        if not path.is_relative_to(raw_dir):
            raise ValueError("Derivative path leaves the manifest directory")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["stream_labels"], STREAMS)
            np.testing.assert_array_equal(saved["arm_labels"], ARMS)
            for order, (data_key, frame_key, _, _, _) in ORDERS.items():
                previous_indices = None
                for version in VERSIONS:
                    prefix = f"{VIEW}__{version}__"
                    indices = saved[prefix + frame_key]
                    if previous_indices is not None:
                        np.testing.assert_array_equal(indices, previous_indices)
                    previous_indices = indices
                    data = saved[prefix + data_key]
                    if data.shape != (len(indices), 2, 2, 6) or data.dtype != np.float64:
                        raise ValueError(f"Invalid full derivative array: {path}/{prefix + data_key}")
                    for stream_id, stream in enumerate(STREAMS):
                        for arm_id, arm in enumerate(ARMS):
                            distributions[order, stream, arm, version].add(data[:, stream_id, arm_id])
        if number % 100 == 0 or number == len(files):
            print(f"Read overlap arrays: {number}/{len(files)} sequences", flush=True)
    for (order, stream, arm, version), distribution in distributions.items():
        distribution.check_summary(stats[version][stream][arm][ORDERS[order][2]])
    return summary, raw_dir, distributions


def draw_panel(ax, distributions, order, stream, arm, tail=False, joint=None):
    unit, threshold = ORDERS[order][3:]
    for version in VERSIONS:
        distribution = distributions[order, stream, arm, version]
        if tail:
            counts = distribution.absolute_counts
            counts = counts.sum(axis=0) if joint is None else counts[joint]
            if counts.sum():
                survival = np.cumsum(counts[::-1])[::-1] / counts.sum()
                included = survival > 0
                ax.plot(distribution.absolute_edges[:-1][included], survival[included],
                        color=COLORS[version], label=LABELS[version], linewidth=1.7)
        else:
            counts = distribution.signed_counts
            counts = counts.sum(axis=0) if joint is None else counts[joint]
            if counts.sum():
                density = counts / (counts.sum() * np.diff(distribution.signed_edges))
                density = np.where(density > 0, density, np.nan)
                ax.stairs(density, distribution.signed_edges,
                          color=COLORS[version], label=LABELS[version], linewidth=1.4)
    ax.set_xscale("symlog", linthresh=threshold)
    ax.set_yscale("log")
    ax.set_xlabel(f"|{order.capitalize()}| threshold ({unit})" if tail else f"{order.capitalize()} ({unit})")
    ax.set_ylabel("Fraction at or above threshold" if tail else "Probability density")
    ax.grid(alpha=0.2, which="major")
    ax.set_axisbelow(True)
    ax.set_title(f"{stream.capitalize()} / {arm} arm" if joint is None else f"Joint {joint + 1}")
    if tail:
        ax.set_xlim(left=0)
        ax.set_ylim(top=1.25)
    else:
        a, b = (distributions[order, stream, arm, version] for version in VERSIONS)
        ra = a.report()["rms"] if joint is None else a.report()["per_joint_rms"][joint]
        rb = b.report()["rms"] if joint is None else b.report()["per_joint_rms"][joint]
        ax.text(0.02, 0.98, f"RMS: {ra:.2f} -> {rb:.2f}", transform=ax.transAxes,
                va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})


def save_figure(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180, facecolor="white")


def make_figures(distributions, output, per_joint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    generated = []
    for order in ORDERS:
        for tail in (False, True):
            stem = f"{order}_{'absolute_tail' if tail else 'signed_distribution'}"
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
            fig.suptitle(f"{order.capitalize()}: Mink vs three-stage / identical overlap intervals", fontsize=14)
            for stream_id, stream in enumerate(STREAMS):
                for arm_id, arm in enumerate(ARMS):
                    draw_panel(axes[stream_id, arm_id], distributions, order, stream, arm, tail)
            axes[0, 0].legend(loc="upper right", fontsize=9)
            save_figure(fig, output, stem)
            plt.close(fig)
            generated.extend([stem + ".png", stem + ".pdf"])
        if per_joint:
            for stream in STREAMS:
                for arm in ARMS:
                    stem = f"{order}_{stream}_{arm}_per_joint"
                    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
                    fig.suptitle(f"{order.capitalize()} / {stream} / {arm}: identical overlap intervals", fontsize=14)
                    for joint, ax in enumerate(axes.flat):
                        draw_panel(ax, distributions, order, stream, arm, joint=joint)
                    axes[0, 0].legend(loc="upper right", fontsize=9)
                    save_figure(fig, output, stem)
                    plt.close(fig)
                    generated.extend([stem + ".png", stem + ".pdf"])
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Completed statistics directory")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <input-dir>/plots_overlap")
    parser.add_argument("--per-joint", action="store_true", help="Also plot six separate joints per stream/arm")
    args = parser.parse_args()
    # Check the plotting dependency before reading the large derivative archives.
    import matplotlib  # noqa: F401

    input_dir = args.input_dir.expanduser().resolve()
    output = (args.output_dir or input_dir / "plots_overlap").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary, raw_dir, distributions = read_distributions(input_dir)
    arrays, reports = {}, {}
    for (order, stream, arm, version), distribution in distributions.items():
        key = f"{order}__{stream}__{arm}__{version}"
        arrays[key + "__signed_edges"] = distribution.signed_edges
        arrays[key + "__signed_counts"] = distribution.signed_counts
        arrays[key + "__absolute_edges"] = distribution.absolute_edges
        arrays[key + "__absolute_counts"] = distribution.absolute_counts
        reports[key] = distribution.report()
    np.savez_compressed(output / "histograms.npz", **arrays)
    figures = make_figures(distributions, output, args.per_joint)
    report = {
        "status": "complete", "view": VIEW, "sequences": summary["sequences"],
        "input_summary": str(input_dir / "summary.json"), "derivatives_directory": str(raw_dir),
        "all_samples_used": True, "sampling": False, "derivatives_recomputed": False,
        "histogram_axes": "Full signed range, symlog x-axis and logarithmic probability density",
        "tail_definition": "Fraction of absolute samples >= each bin lower edge; exact at these thresholds",
        "histogram_rows": "Six rows, one per joint; pooled arm counts are their sum",
        "summary_checks": "Sample counts, RMS, mean absolute, maximum and per-joint RMS match saved summary",
        "distributions": reports, "figures": figures, "seconds": time.monotonic() - started,
    }
    (output / "distribution_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nPLOTS={output}", flush=True)
    for figure in figures:
        if figure.endswith(".png"):
            print(figure)


if __name__ == "__main__":
    main()
