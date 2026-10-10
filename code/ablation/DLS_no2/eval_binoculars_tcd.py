import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from typing import Union
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1


parser = argparse.ArgumentParser()
parser.add_argument("--observer_model_path", type=str, default="./model/Qwen2-0.5B")
parser.add_argument("--performer_model_path", type=str, default="./model/Qwen2-0.5B")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/binoculars')
parser.add_argument('--save_file', type=str, default='eval_binoculars.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def compute_token_weights_from_attention_entropy(
        attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
        attention_mask: torch.Tensor,
        temperature: float = 0.7,
        device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute token weights based on Attention Entropy, consistent with eval_fast_det_gpt_attn_entropy0.py.

    attentions: sequence of layer attentions, each (Batch, Heads, Seq, Seq)
    attention_mask: (Batch, Seq)
    returns: (Batch, Seq-1) weights aligned to positions [1:]
    """
    if attentions is None:
        raise ValueError("Model did not return attentions.")

    first_layer = attentions[0]
    compute_device = (device if device is not None else first_layer.device)
    attn_mask = attention_mask.to(compute_device).float()

    all_layers_entropy: list[torch.Tensor] = []
    layer_means: list[torch.Tensor] = []

    for layer_attn in attentions:
        layer_attn = layer_attn.to(compute_device)
        head_entropy = -torch.sum(layer_attn * torch.log(layer_attn + 1e-10), dim=-1)
        token_entropy = head_entropy.mean(dim=1)
        all_layers_entropy.append(token_entropy)

        per_sample_mean = (token_entropy * attn_mask).sum(dim=-1) / (attn_mask.sum(dim=-1) + 1e-8)
        layer_means.append(per_sample_mean.mean())

    all_layers_mean = torch.stack(layer_means).mean()
    # 仅保留「注意力熵低于所有层平均值」的选层逻辑
    selected_layer_indices = [i for i, m in enumerate(layer_means) if m < all_layers_mean]
    # 兜底：若没有任何层低于均值，则使用全部层
    if len(selected_layer_indices) == 0:
        selected_layer_indices = list(range(len(all_layers_entropy)))

    selected_token_entropy = torch.stack(
        [all_layers_entropy[i] for i in selected_layer_indices], dim=0
    ).mean(dim=0)

    entropy_slice = selected_token_entropy[:, 1:]
    score_for_softmax = -entropy_slice / temperature
    valid_mask = attention_mask[:, 1:].to(compute_device)
    score_for_softmax = score_for_softmax.masked_fill(valid_mask == 0, float('-inf'))
    token_weights = torch.softmax(score_for_softmax, dim=-1)
    return token_weights


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    token_weights: torch.Tensor,
):
    device = logits.device
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_attention_mask = attention_mask[..., 1:].contiguous()

    if shifted_labels.shape[1] == 0:
        return np.zeros((shifted_labels.shape[0],), dtype=np.float32)

    token_nll = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)

    token_weights = token_weights.to(device)
    token_weights = token_weights[:, :token_nll.shape[1]]

    # Weighted aggregation (weights sum to 1 over non-pad tokens)
    token_nll = token_nll * (shifted_attention_mask > 0).to(token_nll.dtype)
    ppl = (token_nll * token_weights).sum(dim=-1)
    return ppl.to("cpu").float().numpy()


def entropy(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        encoding: transformers.BatchEncoding,
        pad_token_id: int,
        token_weights: torch.Tensor,
):
    device = q_logits.device
    attention_mask = encoding.attention_mask.to(device)

    vocab_size = min(p_logits.shape[-1], q_logits.shape[-1])
    p_logits = p_logits[..., :vocab_size].to(device)
    q_logits = q_logits[..., :vocab_size].to(device)

    # Align to next-token positions [:-1] so it matches the shifted labels positions [1:]
    p_scores = p_logits[..., :-1, :].contiguous()
    q_scores = q_logits[..., :-1, :].contiguous()
    total_tokens_available = q_scores.shape[-2]

    if total_tokens_available == 0:
        return np.zeros((q_scores.shape[0],), dtype=np.float32)

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    q_scores_flat = q_scores.view(-1, vocab_size)
    ce = ce_loss_fn(input=q_scores_flat, target=p_proba).view(-1, total_tokens_available)

    token_weights = token_weights.to(device)
    token_weights = token_weights[:, :total_tokens_available]

    agg_ce = (ce * token_weights).sum(dim=-1)
    return agg_ce.to("cpu").float().numpy()


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1).vocab
        == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


# 在当前可见设备内，把 observer 放在第 0 张卡，performer 放在第 1 张卡
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars:
    def __init__(self,
                 observer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B",
                 performer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=dtype,
            output_attentions=False,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=dtype,
            output_attentions=True,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(DEVICE_1)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        observer_outputs = self.observer_model(**encodings.to(DEVICE_1))
        performer_outputs = self.performer_model(**encodings.to(DEVICE_2))
        observer_logits = observer_outputs.logits
        performer_logits = performer_outputs.logits
        performer_attentions = performer_outputs.attentions
        if performer_attentions is None:
            raise ValueError("Performer model did not return attentions. Ensure output_attentions=True.")
        if DEVICE_1.startswith("cuda"):
            torch.cuda.synchronize()
        return observer_logits, performer_logits, performer_attentions

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits, performer_attentions = self._get_logits(encodings)

        token_weights = compute_token_weights_from_attention_entropy(
            attentions=performer_attentions,
            attention_mask=encodings.attention_mask,
            temperature=0.7,
            device=DEVICE_1,
        )

        ppl = perplexity(encodings, performer_logits.to(DEVICE_1), token_weights)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
            token_weights,
        )
        binoculars_scores = - ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Binoculars(
        observer_name_or_path=args.observer_model_path,
        performer_name_or_path=args.performer_model_path,
        use_bfloat16=True,
        max_token_observed=args.max_token_observed,
    )

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    human_scores: list[float] = []
    llm_scores: list[float] = []

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    for idx, item in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating binoculars on {args.eval_data_path.split('/')[-1]}",
    ):
        human_scores.extend(model.compute_score(item['human']))
        llm_scores.extend(model.compute_score(item['llm']))

    # 直接在单进程上计算指标
    
    fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
    prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
    eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)

    
    human_score_mean = float(torch.mean(torch.tensor(human_scores)))
    human_score_std = float(torch.std(torch.tensor(human_scores)))
    llm_score_mean = float(torch.mean(torch.tensor(llm_scores)))
    llm_score_std = float(torch.std(torch.tensor(llm_scores)))

    result_dict = {
        'method': 'binoculars',
        'observer_model_path': args.observer_model_path,
        'performer_model_path': args.performer_model_path,
        'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
        'eval_batch_size': args.eval_batch_size,
        'human_score_mean': human_score_mean,
        'human_score_std': human_score_std,
        'llm_score_mean': llm_score_mean,
        'llm_score_std': llm_score_std,
        'AUROC': eval_auroc,
        'AUPR': eval_aupr,
        'TPR_AT_FPR_5%': tpr_at_5,
        'F1': eval_f1,
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)