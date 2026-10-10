"""Experiment 6 - ABLATE the LAYERS that produce low-entropy patterns.

Reviewer w1: "ablating attention heads/layers that produce low-entropy patterns".

Intervention: Dynamic Layer Selection (DLS) normally keeps the latter-half layers whose
mean attention entropy is below the global average (the low-entropy layers). Here we ABLATE
exactly those layers and instead build the token weights from their COMPLEMENT (the
remaining, higher-entropy layers), keeping the same softmax(-entropy/T) weighting. If the
low-entropy layers are the causal source of the discriminative signal, using the complement
should degrade detection.
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
        # complement=True -> ablate the DLS-selected low-entropy layers, use the rest
        selected = dls_select_token_entropy(layers, attention_mask, complement=True)
        return entropy_to_weights(selected, attention_mask,
                                  temperature=args.temperature, sign=-1.0)

    run_experiment(
        experiment_name="exp6_ablate_low_entropy_layers",
        weight_fn=weight_fn,
        args=args,
        extra_meta={"intervention": "ablate_low_entropy_layers"},
    )


if __name__ == "__main__":
    main()
