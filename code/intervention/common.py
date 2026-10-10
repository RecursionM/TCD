"""Shared machinery for the weight_exp intervention experiments (reviewer w1).

This module provides:
  * three plug-and-play detectors augmented with TCD (Fast-DetectGPT / Binoculars /
    DNA-DetectLLM), each exposing `compute_score(texts, weight_fn)` where `weight_fn`
    is an injectable token-weighting function;
  * helper building blocks for the TCD weighting pipeline (per-layer / per-head
    attention entropy, Dynamic Layer Selection, entropy->weights, token masking,
    weight permutation, head ablation);
  * a `run_experiment(...)` driver that evaluates a given `weight_fn` on all three
    detectors over all datasets and writes result JSONs.

Every experiment file (exp1..exp6) only defines its own `weight_fn` and calls
`run_experiment`, so the intervention logic of each experiment lives in exactly one
place while all detector boilerplate is shared here. The baseline TCD weighting used
by the main paper is `tcd_weight_fn` below.
"""

import os
import json
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import CustomDataset
from metrics import AUROC, AUPR, TPR_at_FPR5, TPR_at_FPR10, F1

torch.set_grad_enabled(False)

# Keep the same two-GPU placement convention as the rest of the repo.
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

# A `weight_fn` maps (attentions, attention_mask) -> token_weights of shape (B, S-1),
# aligned with the next-token positions [1:]. Padding positions must be 0.
WeightFn = Callable[[Union[tuple, list], torch.Tensor], torch.Tensor]


# =============================================================================
# TCD weighting building blocks (all derive purely from attentions + mask)
# =============================================================================
def per_layer_head_entropy(attentions, attention_mask) -> List[torch.Tensor]:
    """Return a list (one per layer) of per-head token attention entropy (B, H, S)."""
    if attentions is None:
        raise ValueError("Model did not return attentions (need output_attentions=True).")
    device = attention_mask.device
    out = []
    for layer_attn in attentions:
        layer_attn = layer_attn.to(device)
        # entropy of each token's attention distribution over keys: (B, H, S)
        head_entropy = -torch.sum(layer_attn * torch.log(layer_attn + 1e-10), dim=-1)
        out.append(head_entropy)
    return out


def per_layer_token_entropy(attentions, attention_mask) -> List[torch.Tensor]:
    """Per-layer token entropy averaged over heads: list of (B, S)."""
    return [he.mean(dim=1) for he in per_layer_head_entropy(attentions, attention_mask)]


def dls_select_token_entropy(
    all_layers_entropy: List[torch.Tensor],
    attention_mask: torch.Tensor,
    complement: bool = False,
) -> torch.Tensor:
    """Dynamic Layer Selection (DLS), identical to the main method.

    Select layers whose mean attention entropy is below the global all-layer mean AND
    that lie in the latter half of the network; average their token entropy.

    If `complement=True`, use the *other* layers instead (i.e. ablate the low-entropy
    layers that DLS would normally keep).
    """
    attn_mask = attention_mask.to(all_layers_entropy[0].device).float()
    layer_means = []
    for token_entropy in all_layers_entropy:
        per_sample_mean = (token_entropy * attn_mask).sum(dim=-1) / (attn_mask.sum(dim=-1) + 1e-8)
        layer_means.append(per_sample_mean.mean())

    all_layers_mean = torch.stack(layer_means).mean()
    mid_layer_idx = (len(layer_means) - 1) // 2
    selected = [
        i for i, m in enumerate(layer_means)
        if (m < all_layers_mean) and (i > mid_layer_idx)
    ]
    if len(selected) == 0:
        selected = [i for i in range(len(all_layers_entropy)) if i > mid_layer_idx]
        if len(selected) == 0:
            selected = list(range(len(all_layers_entropy)))

    if complement:
        comp = [i for i in range(len(all_layers_entropy)) if i not in selected]
        if len(comp) == 0:
            comp = list(range(len(all_layers_entropy)))
        selected = comp

    return torch.stack([all_layers_entropy[i] for i in selected], dim=0).mean(dim=0)


def entropy_to_weights(
    selected_token_entropy: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 0.7,
    sign: float = -1.0,
) -> torch.Tensor:
    """Convert token entropy to softmax weights over positions [1:].

    sign=-1  -> low-entropy tokens get high weight (the TCD default).
    sign=+1  -> high-entropy tokens get high weight (swapped weighting).
    """
    entropy_slice = selected_token_entropy[:, 1:]
    score = sign * entropy_slice / temperature
    valid_mask = attention_mask[:, 1:].to(score.device)
    score = score.masked_fill(valid_mask == 0, float("-inf"))
    return torch.softmax(score, dim=-1)


def uniform_mask_weights(
    selected_token_entropy: torch.Tensor,
    attention_mask: torch.Tensor,
    ratio: float,
    mode: str,
) -> torch.Tensor:
    """Hard-mask a fraction of tokens by entropy, uniform weights over the rest.

    mode="low"  -> drop the `ratio` fraction of lowest-entropy tokens (keep high).
    mode="high" -> drop the `ratio` fraction of highest-entropy tokens (keep low).
    Kept (non-pad) tokens share weight 1/N_keep so this isolates *which tokens are
    used* from *how they are weighted*.
    """
    entropy_slice = selected_token_entropy[:, 1:]
    valid = attention_mask[:, 1:].to(entropy_slice.device).float()
    weights = torch.zeros_like(entropy_slice)
    batch = entropy_slice.shape[0]
    for b in range(batch):
        idx = torch.nonzero(valid[b] > 0, as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            continue
        k = int(round(ratio * n))
        k = max(0, min(k, n - 1))  # always keep at least one token
        order = torch.argsort(entropy_slice[b, idx])  # ascending: low -> high entropy
        if mode == "low":
            drop = order[:k]
        elif mode == "high":
            drop = order[n - k:] if k > 0 else order[:0]
        else:
            raise ValueError(f"Unknown mask mode: {mode}")
        keep_flag = torch.ones(n, dtype=torch.bool, device=entropy_slice.device)
        keep_flag[drop] = False
        keep_idx = idx[keep_flag]
        if keep_idx.numel() == 0:
            keep_idx = idx
        weights[b, keep_idx] = 1.0 / keep_idx.numel()
    return weights


def permute_weights(base_weights: torch.Tensor, attention_mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute the (valid) token weights across positions per sample.

    Weight *values* (and their distribution) are preserved, but the entropy->position
    correspondence is destroyed.
    """
    valid = attention_mask[:, 1:].to(base_weights.device)
    permuted = base_weights.clone()
    gen = torch.Generator()  # CPU generator for reproducible permutations
    gen.manual_seed(int(seed))
    batch = base_weights.shape[0]
    for b in range(batch):
        idx = torch.nonzero(valid[b] > 0, as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n <= 1:
            continue
        perm_local = torch.randperm(n, generator=gen)
        permuted[b, idx] = base_weights[b, idx[perm_local.to(idx.device)]]
    return permuted


def head_ablated_token_entropy(
    attentions,
    attention_mask,
    ratio: float,
    mode: str,
) -> List[torch.Tensor]:
    """Per-layer token entropy after ablating a fraction of attention heads.

    Heads are ranked (per layer) by their mean attention entropy over valid tokens.
    mode="low"  -> drop the lowest-entropy heads (the heads that produce the
                   low-entropy patterns the method relies on).
    mode="high" -> drop the highest-entropy heads (control).
    Remaining heads are averaged to form each layer's token entropy.
    """
    device = attention_mask.device
    attn_mask = attention_mask.to(device).float()  # (B, S)
    per_head = per_layer_head_entropy(attentions, attention_mask)  # list of (B, H, S)
    out = []
    for head_entropy in per_head:
        num_heads = head_entropy.shape[1]
        mask_exp = attn_mask.unsqueeze(1)  # (B, 1, S)
        head_mean = (head_entropy * mask_exp).sum(dim=-1) / (mask_exp.sum(dim=-1) + 1e-8)  # (B, H)
        head_mean = head_mean.mean(dim=0)  # (H,)
        k = int(round(ratio * num_heads))
        k = max(0, min(k, num_heads - 1))
        order = torch.argsort(head_mean)  # ascending: low -> high entropy heads
        if mode == "low":
            drop = order[:k]
        elif mode == "high":
            drop = order[num_heads - k:] if k > 0 else order[:0]
        else:
            raise ValueError(f"Unknown head mode: {mode}")
        keep_flag = torch.ones(num_heads, dtype=torch.bool, device=device)
        keep_flag[drop] = False
        if keep_flag.sum() == 0:
            keep_flag[:] = True
        out.append(head_entropy[:, keep_flag, :].mean(dim=1))  # (B, S)
    return out


def tcd_weight_fn(attentions, attention_mask, temperature: float = 0.7) -> torch.Tensor:
    """The unmodified TCD weighting used by the main paper (reference/baseline)."""
    layers = per_layer_token_entropy(attentions, attention_mask)
    selected = dls_select_token_entropy(layers, attention_mask)
    return entropy_to_weights(selected, attention_mask, temperature=temperature, sign=-1.0)


# =============================================================================
# Detector 1: Fast-DetectGPT (+TCD)
# =============================================================================
def _from_pretrained(cls, model_name, kwargs):
    return cls.from_pretrained(model_name, **kwargs)


def _fast_discrepancy_analytic(logits_ref, logits_score, labels, token_weights):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    weights = token_weights[:, :log_likelihood.shape[1]]
    log_likelihood = (log_likelihood * weights).sum(dim=-1)
    mean_ref = (mean_ref * weights).sum(dim=-1)
    var_ref = (var_ref * (weights ** 2)).sum(dim=-1)

    discrepancy = (log_likelihood - mean_ref) / (var_ref.sqrt() + 1e-8)
    return discrepancy


class FastDetectGPT(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, max_token_observed: int = 512):
        super().__init__()
        self.max_token_observed = max_token_observed
        self.scoring_model_name = scoring_model_name
        self.scoring_model = _from_pretrained(
            AutoModelForCausalLM, scoring_model_name,
            kwargs=dict(torch_dtype=torch.bfloat16, output_attentions=True),
        )
        self.scoring_model.to(DEVICE_1)
        self.scoring_model.eval()
        self.scoring_tokenizer = _from_pretrained(
            AutoTokenizer, scoring_model_name,
            kwargs={"padding_side": "right",
                    "use_fast": "facebook/opt-" not in scoring_model_name},
        )
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        if reference_model_name is not None and reference_model_name != scoring_model_name:
            self.reference_model = _from_pretrained(
                AutoModelForCausalLM, reference_model_name,
                kwargs=dict(torch_dtype=torch.bfloat16),
            )
            self.reference_model.to(DEVICE_2)
            self.reference_model.eval()
            self.reference_tokenizer = _from_pretrained(
                AutoTokenizer, reference_model_name,
                kwargs={"padding_side": "right",
                        "use_fast": "facebook/opt-" not in reference_model_name},
            )
            self.reference_model_name = reference_model_name
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id
        else:
            self.reference_model = None
            self.reference_tokenizer = None
            self.reference_model_name = self.scoring_model_name

    def compute_score(self, input_text, weight_fn: WeightFn):
        input_texts = [input_text] if isinstance(input_text, str) else input_text
        with torch.no_grad():
            tokenized_score = self.scoring_tokenizer(
                input_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_token_observed, return_token_type_ids=False,
            ).to(DEVICE_1)
            outputs = self.scoring_model(**tokenized_score)
            labels = tokenized_score.input_ids[:, 1:]
            logits_score = outputs.logits[:, :-1]

            token_weights = weight_fn(outputs.attentions, tokenized_score.attention_mask)

            if self.scoring_model_name == self.reference_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(
                    input_texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_token_observed, return_token_type_ids=False,
                ).to(DEVICE_2)
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]
                min_seq_len = min(logits_ref.shape[1], logits_score.shape[1])
                logits_ref = logits_ref[:, :min_seq_len, :]
                logits_score = logits_score[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len]
                token_weights = token_weights[:, :min_seq_len]
                token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)

            logits_score = logits_score.to(DEVICE_1)
            logits_ref = logits_ref.to(DEVICE_1)
            labels = labels.to(DEVICE_1)
            token_weights = token_weights.to(DEVICE_1)
            scores = _fast_discrepancy_analytic(logits_ref, logits_score, labels, token_weights)
        return scores.tolist()


# =============================================================================
# Detector 2: Binoculars (+TCD)
# =============================================================================
def _bino_perplexity(encoding, logits, token_weights):
    device = logits.device
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_attention_mask = attention_mask[..., 1:].contiguous()
    if shifted_labels.shape[1] == 0:
        return np.zeros((shifted_labels.shape[0],), dtype=np.float32)
    token_nll = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
    token_weights = token_weights.to(device)[:, :token_nll.shape[1]]
    token_nll = token_nll * (shifted_attention_mask > 0).to(token_nll.dtype)
    ppl = (token_nll * token_weights).sum(dim=-1)
    return ppl.to("cpu").float().numpy()


def _bino_entropy(p_logits, q_logits, encoding, token_weights):
    device = q_logits.device
    vocab_size = min(p_logits.shape[-1], q_logits.shape[-1])
    p_logits = p_logits[..., :vocab_size].to(device)
    q_logits = q_logits[..., :vocab_size].to(device)
    p_scores = p_logits[..., :-1, :].contiguous()
    q_scores = q_logits[..., :-1, :].contiguous()
    total_tokens_available = q_scores.shape[-2]
    if total_tokens_available == 0:
        return np.zeros((q_scores.shape[0],), dtype=np.float32)
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    q_scores_flat = q_scores.view(-1, vocab_size)
    ce = ce_loss_fn(input=q_scores_flat, target=p_proba).view(-1, total_tokens_available)
    token_weights = token_weights.to(device)[:, :total_tokens_available]
    agg_ce = (ce * token_weights).sum(dim=-1)
    return agg_ce.to("cpu").float().numpy()


def _assert_tokenizer_consistency(model_id_1, model_id_2):
    identical = (AutoTokenizer.from_pretrained(model_id_1).vocab
                 == AutoTokenizer.from_pretrained(model_id_2).vocab)
    if not identical:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


class Binoculars:
    def __init__(self, observer_name_or_path, performer_name_or_path,
                 use_bfloat16: bool = True, max_token_observed: int = 512):
        _assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path, device_map={"": DEVICE_1},
            trust_remote_code=False, torch_dtype=dtype, output_attentions=False)
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path, device_map={"": DEVICE_2},
            trust_remote_code=False, torch_dtype=dtype, output_attentions=True)
        self.observer_model.eval()
        self.performer_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch):
        return self.tokenizer(
            batch, return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True, max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(DEVICE_1)

    @torch.inference_mode()
    def compute_score(self, input_text, weight_fn: WeightFn):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_outputs = self.performer_model(**encodings.to(DEVICE_2))
        performer_logits = performer_outputs.logits

        token_weights = weight_fn(performer_outputs.attentions, encodings.attention_mask)
        if DEVICE_1.startswith("cuda"):
            torch.cuda.synchronize()

        ppl = _bino_perplexity(encodings, performer_logits.to(DEVICE_1), token_weights)
        x_ppl = _bino_entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                              encodings.to(DEVICE_1), token_weights)
        scores = (-ppl / x_ppl).tolist()
        return scores[0] if isinstance(input_text, str) else scores


# =============================================================================
# Detector 3: DNA-DetectLLM (+TCD)
# =============================================================================
def _dna_sum_perplexity(encoding, logits, token_weights):
    shifted_logits = logits[..., :-1, :]
    attention = encoding.attention_mask[..., 1:].to(shifted_logits.device)
    labels_std = encoding.input_ids[..., 1:].to(shifted_logits.device)
    labels_max = torch.argmax(shifted_logits, dim=-1)
    logits_T = shifted_logits.transpose(1, 2)
    ce_std = F.cross_entropy(logits_T, labels_std, reduction="none")
    ce_max = F.cross_entropy(logits_T, labels_max, reduction="none")
    token_scores = ce_std + ce_max
    weights = token_weights.to(token_scores.device)[..., :token_scores.shape[1]]
    weights = weights * attention
    denom = weights.sum(dim=1).clamp(min=1e-8)
    ppl = (token_scores * weights).sum(dim=1) / denom
    return ppl.float().cpu().numpy()


def _dna_entropy(p_logits, q_logits, encoding, token_weights):
    p_scores = p_logits[:, :-1, :]
    q_scores = q_logits[:, :-1, :]
    p_probs = F.softmax(p_scores, dim=-1)
    q_log_probs = F.log_softmax(q_scores, dim=-1)
    token_ce = -(p_probs * q_log_probs).sum(dim=-1)
    padding_mask = encoding.attention_mask[:, 1:].to(token_ce.device).float()
    weights = token_weights.to(token_ce.device)[..., :token_ce.shape[1]]
    weights = weights * padding_mask
    denom = weights.sum(dim=1).clamp(min=1e-8)
    agg_ce = (token_ce * weights).sum(dim=1) / denom
    return agg_ce.detach().cpu().float().numpy()


class DetectLLM:
    def __init__(self, observer_name_or_path, performer_name_or_path,
                 use_bfloat16: bool = True, max_token_observed: int = 512):
        _assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path, device_map={"": DEVICE_1},
            trust_remote_code=True, torch_dtype=dtype)
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path, device_map={"": DEVICE_2},
            trust_remote_code=True, torch_dtype=dtype, output_attentions=True)
        self.observer_model.eval()
        self.performer_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch):
        return self.tokenizer(
            batch, return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True, max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)

    @torch.inference_mode()
    def compute_score(self, input_text, weight_fn: WeightFn):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_outputs = self.performer_model(**encodings.to(DEVICE_2), output_attentions=True)
        performer_logits = performer_outputs.logits

        token_weights = weight_fn(performer_outputs.attentions, encodings.attention_mask)

        ppl = _dna_sum_perplexity(encodings, performer_logits, token_weights)
        x_ppl = _dna_entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                             encodings.to(DEVICE_1), token_weights.to(DEVICE_1))
        scores = (ppl / (2 * x_ppl)).tolist()
        return scores


# =============================================================================
# Shared experiment driver
# =============================================================================
DETECTOR_SPECS = {
    "Fast-DetectGPT_tcd": {"prefix": "Fast-DetectGPT_tcd", "negate": False},
    "Binoculars_tcd": {"prefix": "Bino_tcd", "negate": False},
    "DNA-DetectLLM_tcd": {"prefix": "dnadetectllm_tcd", "negate": True},
}


def _build_detector(name, model_base, model_instruct, max_token_observed):
    if name == "Fast-DetectGPT_tcd":
        return FastDetectGPT(scoring_model_name=model_instruct,
                             reference_model_name=model_base,
                             max_token_observed=max_token_observed)
    if name == "Binoculars_tcd":
        return Binoculars(observer_name_or_path=model_base,
                          performer_name_or_path=model_instruct,
                          max_token_observed=max_token_observed)
    if name == "DNA-DetectLLM_tcd":
        return DetectLLM(observer_name_or_path=model_base,
                         performer_name_or_path=model_instruct,
                         max_token_observed=max_token_observed)
    raise ValueError(f"Unknown detector: {name}")


def _evaluate_one(detector, spec, dataset_path, data_format, batch_size, weight_fn):
    dataset = CustomDataset(data_path=dataset_path, data_format=data_format)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    human_scores, llm_scores = [], []
    from tqdm import tqdm
    for _, item in tqdm(enumerate(loader), total=len(loader),
                        desc=f"{spec['prefix']} on {os.path.basename(dataset_path)}"):
        h = detector.compute_score(item["human"], weight_fn)
        l = detector.compute_score(item["llm"], weight_fn)
        if spec["negate"]:
            h = [-s for s in h]
            l = [-s for s in l]
        human_scores.extend(h)
        llm_scores.extend(l)
    return human_scores, llm_scores


def run_experiment(experiment_name, weight_fn, args, extra_meta: Optional[dict] = None):
    """Evaluate a single `weight_fn` on all detectors and datasets, writing JSONs.

    Layout mirrors the repo convention:
      <save_root>/<experiment_name>/<Detector>/<prefix>_<dataset>.json
    """
    save_root = os.path.join(args.save_root, experiment_name)
    detectors_to_run = args.detectors if args.detectors else list(DETECTOR_SPECS.keys())

    for det_name in detectors_to_run:
        spec = DETECTOR_SPECS[det_name]
        detector = _build_detector(det_name, args.model_base, args.model_instruct,
                                    args.max_token_observed)
        out_dir = os.path.join(save_root, det_name)
        os.makedirs(out_dir, exist_ok=True)
        for file_name in args.datasets:
            dataset_path = os.path.join(args.data_dir, file_name)
            clean_name = os.path.basename(file_name)[:-len(".json")] if file_name.endswith(".json") else file_name
            human_scores, llm_scores = _evaluate_one(
                detector, spec, dataset_path, args.eval_data_format,
                args.eval_batch_size, weight_fn)

            _, _, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
            _, _, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
            tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
            tpr_at_10 = TPR_at_FPR10(neg_list=human_scores, pos_list=llm_scores)
            eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)
            result = {
                "experiment": experiment_name,
                "detector": det_name,
                "model_base": args.model_base,
                "model_instruct": args.model_instruct,
                "eval_dataset": clean_name,
                "eval_batch_size": args.eval_batch_size,
                "AUROC": eval_auroc,
                "AUPR": eval_aupr,
                "TPR_AT_FPR_5%": tpr_at_5,
                "TPR_AT_FPR_10%": tpr_at_10,
                "F1": eval_f1,
                "human_score_mean": float(np.mean(human_scores)) if human_scores else 0.0,
                "llm_score_mean": float(np.mean(llm_scores)) if llm_scores else 0.0,
            }
            if extra_meta:
                result.update(extra_meta)
            out_file = os.path.join(out_dir, f"{spec['prefix']}_{clean_name}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, indent=4))
            print(f"[{experiment_name}] {det_name} {clean_name}: AUROC={eval_auroc:.4f}")

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_common_parser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data_dir", type=str, default="datasets/Benchmarks_1k")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["RealDet.json", "Raid.json", "M4.json"])
    p.add_argument("--detectors", type=str, nargs="*", default=None,
                   help="Subset of detectors to run; default runs all three.")
    p.add_argument("--eval_data_format", type=str, default="ImBD")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_token_observed", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_base", type=str,
                   default="meta-llama/Llama-3.2-1B")
    p.add_argument("--model_instruct", type=str,
                   default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--save_root", type=str, default="./results_weight_exp")
    return p
