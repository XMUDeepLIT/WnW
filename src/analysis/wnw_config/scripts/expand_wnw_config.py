
import argparse
import json


def expand_config(cfg: dict, num_layers: int, orig_kv_heads: int, num_q_heads: int) -> dict:
    groups = num_q_heads // orig_kv_heads
    assert num_q_heads % orig_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by orig_kv_heads ({orig_kv_heads})"
    )

    rr = cfg["heads"]["retention_ratios"]
    assert len(rr) == num_layers * orig_kv_heads, (
        f"retention_ratios length {len(rr)} != num_layers * orig_kv_heads ({num_layers * orig_kv_heads})"
    )

    expanded_rr = []
    for layer in range(num_layers):
        for kv_h in range(orig_kv_heads):
            ratio = rr[layer * orig_kv_heads + kv_h]
            expanded_rr.extend([ratio] * groups)

    expanded_cb = []
    for gid in cfg["heads"]["class_b"]:
        layer_idx = gid // orig_kv_heads
        local_kv = gid % orig_kv_heads
        base_q = layer_idx * num_q_heads + local_kv * groups
        expanded_cb.extend(range(base_q, base_q + groups))

    result = dict(cfg)
    result["heads"] = dict(cfg["heads"])
    result["heads"]["retention_ratios"] = expanded_rr
    result["heads"]["class_b"] = expanded_cb
    result["heads"]["_orig_kv_heads"] = orig_kv_heads
    result["heads"]["_expanded_from_gqa"] = True

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--orig_kv_heads", type=int, required=True)
    parser.add_argument("--num_q_heads", type=int, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        cfg = json.load(f)

    expanded = expand_config(cfg, args.num_layers, args.orig_kv_heads, args.num_q_heads)

    with open(args.output, "w") as f:
        json.dump(expanded, f, indent=2)

    rr = expanded["heads"]["retention_ratios"]
    cb = expanded["heads"]["class_b"]
    ca = expanded["heads"]["class_a"]
    print(f"Original: {args.num_layers} layers x {args.orig_kv_heads} kv_heads = {args.num_layers * args.orig_kv_heads}")
    print(f"Expanded: {args.num_layers} layers x {args.num_q_heads} kv_heads = {args.num_layers * args.num_q_heads}")
    print(f"retention_ratios: {len(rr)} (expected {args.num_layers * args.num_q_heads})")
    print(f"class_a: {len(ca)} q_heads (unchanged)")
    print(f"class_b: {len(cb)} q_heads (expanded from {len(cfg['heads']['class_b'])} kv_heads)")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
