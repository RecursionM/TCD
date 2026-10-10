"""Experiment 4 - SWAP entropy-based weights (invert the direction).

Reviewer w1: "randomizing or swapping entropy-based weights".

Intervention: keep the whole TCD pipeline (per-head entropy -> DLS layer selection) but
flip the sign of the weighting, i.e. use softmax(+entropy/T) instead of softmax(-entropy/T)
so that HIGH-entropy tokens receive the large weights. If the *direction* of the weighting
is what drives detection, this swapped variant should perform worse than uniform weighting.
(This mirrors the existing `ablation_method/inverse_reweight`, reproduced here so all six
w1 interventions live under one folder.)
"""
from common import (
    build_common_parser, run_experiment,
    per_layer_token_entropy, dls_select_token_entropy, entropy_to_weights,
)


def main():
    parser = build_common_parser(__doc__)
    args = parser.parse_args()

    def weight_fn(attentions, attention_mask):
        layers = per_layer_token_entropy(attentions, attention_mask)
        selected = dls_select_token_entropy(layers, attention_mask)
        # sign=+1 -> high-entropy tokens get the high weights (swapped)
        return entropy_to_weights(selected, attention_mask,
                                  temperature=args.temperature, sign=+1.0)

    run_experiment(
        experiment_name="exp4_swap_weights",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "swap_invert_weights"},
    )


if __name__ == "__main__":
    main()
