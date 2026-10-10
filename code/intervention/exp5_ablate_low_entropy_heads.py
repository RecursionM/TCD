"""Experiment 5 - ABLATE attention HEADS that produce low-entropy patterns.

Reviewer w1: "ablating attention heads/layers that produce low-entropy patterns".

Intervention: within every layer, rank attention heads by their mean attention entropy and
DROP the `head_ratio` fraction with the LOWEST entropy (the heads that generate the focused,
low-entropy patterns the method exploits). Token entropy is then averaged over the remaining
heads before the usual DLS selection and softmax(-entropy/T) weighting. If those heads are
the causal source of the signal, removing them should degrade detection.
"""
from common import (
    build_common_parser, run_experiment,
    head_ablated_token_entropy, dls_select_token_entropy, entropy_to_weights,
)


def main():
    parser = build_common_parser(__doc__)
    parser.add_argument("--head_ratio", type=float, default=0.3,
                        help="Fraction of lowest-entropy heads to ablate per layer.")
    args = parser.parse_args()

    def weight_fn(attentions, attention_mask):
        layers = head_ablated_token_entropy(attentions, attention_mask,
                                            ratio=args.head_ratio, mode="low")
        selected = dls_select_token_entropy(layers, attention_mask)
        return entropy_to_weights(selected, attention_mask,
                                  temperature=args.temperature, sign=-1.0)

    run_experiment(
        experiment_name="exp5_ablate_low_entropy_heads",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "ablate_low_entropy_heads", "head_ratio": args.head_ratio},
    )


if __name__ == "__main__":
    main()
