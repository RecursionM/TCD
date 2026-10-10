"""Experiment 2 - Mask HIGH-entropy tokens.

Reviewer w1: "masking low- vs. high-entropy tokens".

Intervention: identical pipeline to exp1, but HARD-MASK the `mask_ratio` fraction of the
highest-entropy tokens (the ones the method already down-weights) and aggregate the rest
with *uniform* weights. If the low-entropy tokens carry the signal, removing high-entropy
tokens should barely hurt (or even help) detection. The gap between exp1 (mask low) and
exp2 (mask high) is the causal evidence the reviewer asks for.
"""
from common import (
    build_common_parser, run_experiment,
    per_layer_token_entropy, dls_select_token_entropy, uniform_mask_weights,
)


def main():
    parser = build_common_parser(__doc__)
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="Fraction of highest-entropy tokens to drop.")
    args = parser.parse_args()

    def weight_fn(attentions, attention_mask):
        layers = per_layer_token_entropy(attentions, attention_mask)
        selected = dls_select_token_entropy(layers, attention_mask)
        return uniform_mask_weights(selected, attention_mask, ratio=args.mask_ratio, mode="high")

    run_experiment(
        experiment_name="exp2_mask_high_entropy",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "mask_high_entropy_tokens", "mask_ratio": args.mask_ratio},
    )


if __name__ == "__main__":
    main()
