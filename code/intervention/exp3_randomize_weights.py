"""Experiment 3 - RANDOMIZE entropy-based weights.

Reviewer w1: "randomizing or swapping entropy-based weights".

Intervention: compute the exact TCD token weights, then randomly PERMUTE them across the
(valid) token positions of each sample. The weight *distribution* is preserved, but the
entropy->position correspondence is destroyed. If the correspondence (not just the weight
shape) is what matters, performance should collapse toward uniform weighting.
"""
from common import (
    build_common_parser, run_experiment,
    tcd_weight_fn, permute_weights,
)


def main():
    parser = build_common_parser(__doc__)
    args = parser.parse_args()

    def weight_fn(attentions, attention_mask):
        base = tcd_weight_fn(attentions, attention_mask, temperature=args.temperature)
        return permute_weights(base, attention_mask, seed=args.seed)

    run_experiment(
        experiment_name="exp3_randomize_weights",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "randomize_permute_weights", "seed": args.seed},
    )


if __name__ == "__main__":
    main()
