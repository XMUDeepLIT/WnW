
import argparse
import json
import os
import subprocess
import sys

from common.adapters import get_adapter

NUM_CALIB_SAMPLES = 3
TOLERANCE = 0.02
PYTHON = sys.executable

_GEN_SCRIPT = "src/analysis/wnw_config/scripts/generate_wnw_config.py"
_EVAL_SCRIPT = "src/benchmarks/scripts/eval_wnw.py"


def _run(cmd):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
    else:
        print(result.stdout)
    return result


def _parse_estimated_scale(stdout):
    for line in stdout.splitlines():
        if "retention_scale=" in line:
            try:
                return float(line.split("retention_scale=")[-1].split()[0])
            except ValueError:
                continue
    return None


def make_3sample_input(eval_input, output_dir):
    subset_path = os.path.join(output_dir, "calib_3samples.jsonl")
    with open(eval_input) as fin, open(subset_path, "w") as fout:
        for i, line in enumerate(fin):
            if i >= NUM_CALIB_SAMPLES:
                break
            fout.write(line)
    return subset_path


def _base_cmd(head_scores_path, output_dir, token_rate, num_class_a,
              sensitivity_mode, top_voice, duration_file=None, loadback_factor=0.8):
    config_path = os.path.join(output_dir, "config.json")
    cmd = (
        f"PYTHONPATH=src {PYTHON} {_GEN_SCRIPT} "
        f"--head-scores-path {head_scores_path} --output {config_path} "
        f"--token-rate {token_rate} --num-class-a {num_class_a} "
        f"--sensitivity-mode {sensitivity_mode} "
        f"--top-voice {top_voice} --chunk-size-s 4.0 --overlap-s 2.0 "
        f"--loadback-factor {loadback_factor}"
    )
    if duration_file:
        cmd += f" --duration-file {duration_file}"
    return cmd, config_path


def generate_config_target(head_scores_path, target, output_dir, token_rate,
                           num_class_a, sensitivity_mode, top_voice,
                           duration_file=None, loadback_factor=0.8):
    """Generate config by asking generate_wnw_config.py to bisect for the target.

    Returns (config_path, estimated_scale) or (None, None) on failure.
    """
    cmd, config_path = _base_cmd(
        head_scores_path, output_dir, token_rate, num_class_a,
        sensitivity_mode, top_voice, duration_file, loadback_factor,
    )
    cmd += f" --target-gpucpu-retention {target:.6f}"
    result = _run(cmd)
    if result.returncode != 0:
        return None, None
    return config_path, _parse_estimated_scale(result.stdout)


def generate_config_scale(head_scores_path, scale, output_dir, token_rate,
                          num_class_a, sensitivity_mode, top_voice,
                          duration_file=None, loadback_factor=0.8):
    """Generate config with an explicit retention_scale."""
    cmd, config_path = _base_cmd(
        head_scores_path, output_dir, token_rate, num_class_a,
        sensitivity_mode, top_voice, duration_file, loadback_factor,
    )
    cmd += f" --retention-scale {scale:.6f}"
    return config_path if _run(cmd).returncode == 0 else None


def run_eval(config_path, eval_input, output_dir, model_path, device, adapter_name):
    result_path = os.path.join(output_dir, "calib_result.jsonl")
    if os.path.exists(result_path):
        os.remove(result_path)
    cmd = (
        f"PYTHONPATH=src {PYTHON} {_EVAL_SCRIPT} "
        f"--input {eval_input} --config {config_path} --output {result_path} "
        f"--model-path {model_path} --adapter {adapter_name} --device {device} "
        f"--max-new-tokens 200"
    )
    if _run(cmd).returncode != 0:
        return None
    retentions = []
    with open(result_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            snaps = obj.get("stats", {}).get("decode_snapshots", [])
            df = snaps[-1] if snaps else obj.get("stats", {}).get("prefill", {})
            r = df.get("gpu_plus_cpu_retention")
            if r is not None:
                retentions.append(r)
    return sum(retentions) / len(retentions) if retentions else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-scores-path", type=str, required=True)
    parser.add_argument("--target-retention", type=float, required=True)
    parser.add_argument("--eval-input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter", type=str, default="voxtral",
                        choices=["voxtral", "qwen2_5_omni"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-class-a", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--sensitivity-mode", type=str, default="mean",
                        choices=["k", "v", "mean"])
    parser.add_argument("--token-rate", type=float, default=None)
    parser.add_argument("--top-voice", type=float, default=0.10)
    parser.add_argument("--duration-file", type=str, default=None,
                        help="File with one float per line (duration in seconds)")
    parser.add_argument("--loadback-factor", type=float, default=0.8,
                        help="Empirical damping on CPU->GPU loadback (default: 0.8)")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)
    if args.token_rate is None:
        args.token_rate = adapter.token_rate

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== Calibration for target_retention={args.target_retention} ===")

    subset_path = make_3sample_input(args.eval_input, args.output_dir)
    print(f"Calibration subset: {subset_path} ({NUM_CALIB_SAMPLES} samples)")

    config_path, est_scale = generate_config_target(
        args.head_scores_path, args.target_retention, args.output_dir, args.token_rate,
        args.num_class_a, args.sensitivity_mode, args.top_voice,
        duration_file=args.duration_file, loadback_factor=args.loadback_factor,
    )
    if config_path is None:
        print("Failed to generate initial config")
        return
    if est_scale is None:
        est_scale = 0.15
        print(f"Could not parse estimated scale; falling back to {est_scale}")

    actual_ret = run_eval(config_path, subset_path, args.output_dir, args.model_path, args.device, args.adapter)
    if actual_ret is None:
        print("Failed to get initial measurement")
        return

    print(f"Initial: scale={est_scale:.4f} -> retention={actual_ret:.4f} (target={args.target_retention})")

    best_scale, best_ret = est_scale, actual_ret
    best_diff = abs(actual_ret - args.target_retention)
    if best_diff <= args.tolerance:
        print(f"Within tolerance ({args.tolerance}). Final scale={best_scale:.6f}\nConfig: {config_path}")
        return

    for round_idx, (lo_mul, hi_mul, n_steps) in enumerate([(0.3, 3.0, 8), (0.7, 1.3, 6)]):
        base = best_scale
        lo, hi = base * lo_mul, base * hi_mul
        scales = [lo + (hi - lo) * i / (n_steps - 1) for i in range(n_steps)]
        print(f"\nRound {round_idx + 1}: scanning {n_steps} scales in [{lo:.4f}, {hi:.4f}]")
        for scale in scales:
            cp = generate_config_scale(
                args.head_scores_path, scale, args.output_dir, args.token_rate,
                args.num_class_a, args.sensitivity_mode, args.top_voice,
                duration_file=args.duration_file, loadback_factor=args.loadback_factor,
            )
            if cp is None:
                continue
            r = run_eval(cp, subset_path, args.output_dir, args.model_path, args.device, args.adapter)
            if r is None:
                continue
            diff = abs(r - args.target_retention)
            print(f"  scale={scale:.4f} -> retention={r:.4f} (diff={diff:.4f})")
            if diff < best_diff:
                best_diff, best_scale, best_ret = diff, scale, r
            if diff <= args.tolerance:
                break
        if best_diff <= args.tolerance:
            break

    print(f"\n=== Result ===")
    print(f"Best scale: {best_scale:.6f}")
    print(f"Actual retention: {best_ret:.4f} (target={args.target_retention}, diff={best_diff:.4f})")
    config_path = generate_config_scale(
        args.head_scores_path, best_scale, args.output_dir, args.token_rate,
        args.num_class_a, args.sensitivity_mode, args.top_voice,
        duration_file=args.duration_file, loadback_factor=args.loadback_factor,
    )
    print(f"Final config: {config_path}")


if __name__ == "__main__":
    main()
