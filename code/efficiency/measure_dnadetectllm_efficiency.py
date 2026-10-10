import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from dataset import CustomDataset
from dna_detectllm import DetectLLM
from efficiency_utils import SampleEfficiency, measure_one_text, save_efficiency_json, summarize_efficiency


torch.set_grad_enabled(False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure per-sample time/memory for DNA-DetectLLM.")
    p.add_argument("--observer_name_or_path", type=str, required=True)
    p.add_argument("--performer_name_or_path", type=str, required=True)

    p.add_argument(
        "--eval_data_path",
        type=str,
        default="datasets/Benchmarks_1k/Raid_gt300_balanced_1000_llama3.2-1B-Instruct.json",
    )
    p.add_argument("--eval_data_format", type=str, default="ImBD")
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_token_observed", type=int, default=300)

    p.add_argument("--save_path", type=str, default="results_efficiency/DNA-DetectLLM")
    p.add_argument("--save_file", type=str, default=None)
    return p


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("This efficiency script requires --eval_batch_size=1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    detector = DetectLLM(
        observer_name_or_path=args.observer_name_or_path,
        performer_name_or_path=args.performer_name_or_path,
        max_token_observed=args.max_token_observed,
    )

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    dataset_name = os.path.basename(args.eval_data_path).replace(".json", "")
    save_file = args.save_file or f"efficiency_dnadetectllm_{dataset_name}.json"
    out_path = os.path.join(args.save_path, save_file)

    rows: List[SampleEfficiency] = []

    total_texts = len(dataset) * 2
    pbar = tqdm(total=total_texts, desc=f"Measuring DNA-DetectLLM on {dataset_name} (1 text = 1 sample)")
    sample_idx = 0

    for group_idx, item in enumerate(data_loader):
        rows.append(
            measure_one_text(
                sample_idx=sample_idx,
                text_fn=lambda: detector.compute_score(item["human"]),
                source="human",
                group_idx=int(group_idx),
            )
        )
        sample_idx += 1
        pbar.update(1)

        rows.append(
            measure_one_text(
                sample_idx=sample_idx,
                text_fn=lambda: detector.compute_score(item["llm"]),
                source="llm",
                group_idx=int(group_idx),
            )
        )
        sample_idx += 1
        pbar.update(1)

    pbar.close()

    summary = summarize_efficiency(rows)

    payload: Dict[str, Any] = {
        "method": "DNA-DetectLLM",
        "sample_definition": "one text = one sample",
        "eval_data_path": args.eval_data_path,
        "eval_data_format": args.eval_data_format,
        "eval_batch_size": 1,
        "max_token_observed": args.max_token_observed,
        "observer_name_or_path": args.observer_name_or_path,
        "performer_name_or_path": args.performer_name_or_path,
        "efficiency_summary": summary,
        "per_sample": [r.__dict__ for r in rows],
    }

    save_efficiency_json(out_path, payload)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
