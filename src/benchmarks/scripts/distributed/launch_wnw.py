
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _eval_script():
    return Path(__file__).resolve().parent.parent / "eval_wnw.py"


def _parse_gpus(gpus_str):
    if "," in gpus_str:
        return [int(x.strip()) for x in gpus_str.split(",") if x.strip()]
    n = int(gpus_str)
    return list(range(n))


def _shard_samples(samples, n_shards):
    shards = [[] for _ in range(n_shards)]
    for i, s in enumerate(samples):
        shards[i % n_shards].append(s)
    return shards


def main():
    parser = argparse.ArgumentParser(description="Distributed WnW evaluation launcher")
    parser.add_argument("--gpus", type=str, default="8",
                        help="GPU IDs to use, e.g. '0,1,2,3' or '8' for 0..7")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter", type=str, default="voxtral")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--stats-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--degqa", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    gpus = _parse_gpus(args.gpus)
    n_gpus = len(gpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    with open(args.input) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Total samples: {len(samples)}, sharding across {n_gpus} GPUs")

    shards = _shard_samples(samples, n_gpus)
    for i, shard in enumerate(shards):
        shard_file = shard_dir / f"shard_{i}.jsonl"
        with open(shard_file, "w") as f:
            for s in shard:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    eval_script = _eval_script()
    processes = []
    for idx, gpu_id in enumerate(gpus):
        shard_file = shard_dir / f"shard_{idx}.jsonl"
        output_file = output_dir / f"shard_{gpu_id}.jsonl"
        cmd = [
            args.python, str(eval_script),
            "--input", str(shard_file),
            "--output", str(output_file),
            "--config", args.config,
            "--model-path", args.model_path,
            "--adapter", args.adapter,
            "--device", f"cuda:0",
            "--max-new-tokens", str(args.max_new_tokens),
            "--sample-rate", str(args.sample_rate),
            "--stats-interval", str(args.stats_interval),
            "--batch-size", str(args.batch_size),
        ]
        if args.degqa:
            cmd.append("--degqa")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"  GPU {gpu_id}: {len(shards[idx])} samples -> {output_file}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((gpu_id, proc))

    print(f"\nAll {n_gpus} workers launched. Waiting...")
    t0 = time.time()
    failed = []
    for gpu_id, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()[-500:]
            print(f"  GPU {gpu_id} FAILED (rc={proc.returncode}): {stderr}")
            failed.append(gpu_id)
        else:
            print(f"  GPU {gpu_id} done.")

    elapsed = time.time() - t0
    print(f"\nAll workers finished in {elapsed/60:.1f} minutes. Failed: {failed}")

    merged = output_dir / "merged.jsonl"
    results = []
    for gpu_id in gpus:
        shard_output = output_dir / f"shard_{gpu_id}.jsonl"
        if shard_output.exists():
            with open(shard_output) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
    results.sort(key=lambda x: x["id"])
    with open(merged, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(results)} results -> {merged}")


if __name__ == "__main__":
    main()
