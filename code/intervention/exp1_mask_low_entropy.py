"""Experiment 1 - Mask LOW-entropy tokens.

Reviewer w1: "masking low- vs. high-entropy tokens".

Intervention: run the normal TCD pipeline (per-head entropy -> DLS layer selection),
then HARD-MASK the `mask_ratio` fraction of the lowest-entropy tokens and aggregate the
remaining tokens with *uniform* weights. This removes exactly the tokens the method
claims are most informative. If low-entropy tokens causally drive detection, AUROC
should drop sharply here (to be contrasted with exp2, which masks high-entropy tokens).
"""
from common import (
    build_common_parser, run_experiment,
    per_layer_token_entropy, dls_select_token_entropy, uniform_mask_weights,
)


def main():
    parser = build_common_parser(__doc__)
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="Fraction of lowest-entropy tokens to drop.")
    args = parser.parse_args()

    def weight_fn(attentions, attention_mask):
        layers = per_layer_token_entropy(attentions, attention_mask)
        selected = dls_select_token_entropy(layers, attention_mask)
        return uniform_mask_weights(selected, attention_mask, ratio=args.mask_ratio, mode="low")

    run_experiment(
        experiment_name="exp1_mask_low_entropy",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "mask_low_entropy_tokens", "mask_ratio": args.mask_ratio},
    )


if __name__ == "__main__":
    main()
