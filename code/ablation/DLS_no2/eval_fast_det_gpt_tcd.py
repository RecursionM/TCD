import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1
import torch.nn as nn
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen/Qwen2-0.5B")
parser.add_argument("--reference_model_name", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--discrepancy_analytic", action="store_true", help="Whether to use the analytic version of sampling discrepancy.")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_fast_detect_gpt.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def get_samples(logits, labels):
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels, token_weights=None):
    """
    计算加权 Likelihood。
    token_weights: (Batch, Seq) 已经归一化过的权重
    """
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)

    if token_weights is not None:
        # 确保权重是对齐的
        weights = token_weights[:, :log_likelihood.shape[1]]
        # 加权求和 (weights sum=1)
        log_likelihood = (log_likelihood * weights).sum(dim=-1)
    else:
        log_likelihood = log_likelihood.mean(dim=-1)
        
    return log_likelihood

def get_sampling_discrepancy(logits_ref, logits_score, labels, token_weights=None):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)

    # 原始文本的 Likelihood (加权)
    log_likelihood_x = get_likelihood(logits_score, labels, token_weights)
    # 采样文本的 Likelihood (加权) - 使用相同的权重，假设重要性由位置决定
    log_likelihood_x_tilde = get_likelihood(logits_score, samples, token_weights)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x - miu_tilde) / (sigma_tilde + 1e-8)
    return discrepancy

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, token_weights=None):
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

    if token_weights is not None:
        weights = token_weights[:, :log_likelihood.shape[1]]

        # 1. 加权均值：E[Sum(w_i * X_i)] = Sum(w_i * E[X_i])
        log_likelihood = (log_likelihood * weights).sum(dim=-1)
        mean_ref = (mean_ref * weights).sum(dim=-1)

        # 2. 加权方差：Var(Sum(w_i * X_i)) = Sum(w_i^2 * Var(X_i)) 
        # 注意：这里权重需要平方
        var_ref = (var_ref * (weights ** 2)).sum(dim=-1)
    else:
        log_likelihood = log_likelihood.sum(dim=-1)
        mean_ref = mean_ref.sum(dim=-1)
        var_ref = var_ref.sum(dim=-1)

    discrepancy = (log_likelihood - mean_ref) / (var_ref.sqrt() + 1e-8)
    return discrepancy


def from_pretrained(cls, model_name, kwargs):
    return cls.from_pretrained(model_name, **kwargs)


class FastDetectGPT(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 reference_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,
                 discrepancy_analytic: bool=False,
                 max_token_observed: int = 512,
                 ):
        super().__init__()
        self.cache_dir = cache_dir
        self.max_token_observed = max_token_observed
        
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                 scoring_model_name,
                                                 kwargs=dict(dtype=torch.bfloat16, output_attentions=True))
            self.scoring_model.to(DEVICE_1)
            self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                     scoring_model_name,
                                                     kwargs={'padding_side': 'right',
                                                             'use_fast': True if 'facebook/opt-' not in scoring_model_name else False},
                                                     )
        else: 
            if scoring_model is None or scoring_tokenizer is None:
                raise ValueError('You should provide scoring_model_name or scoring_model and scoring_tokenizer.')
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
            # 强制开启 output_attentions
            self.scoring_model.config.output_attentions = True

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        if reference_model_name is not None:
            if reference_model_name != scoring_model_name:
                self.reference_model = from_pretrained(AutoModelForCausalLM,
                                                        reference_model_name,
                                                        kwargs=dict(dtype=torch.bfloat16))
                self.reference_model.to(DEVICE_2)
                self.reference_tokenizer = from_pretrained(AutoTokenizer,
                                                                reference_model_name,
                                                                kwargs={'padding_side': 'right',
                                                                        'use_fast': True if 'facebook/opt-' not in reference_model_name else False},
                                                                )
                self.reference_model_name = reference_model_name
            else:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
        else:
            if reference_model is None and reference_tokenizer is None:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
            elif reference_model is not None and reference_tokenizer is not None:
                self.reference_model = reference_model
                self.reference_tokenizer = reference_tokenizer
                self.reference_model_name = reference_model.config._name_or_path
            else:
                raise ValueError('You should provide reference_model and reference_tokenizer at the same time.')
        self.discrepancy = get_sampling_discrepancy_analytic if discrepancy_analytic else get_sampling_discrepancy

        if self.reference_tokenizer is not None:
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id
   
    
    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text
        with torch.no_grad():
            tokenized_score = self.scoring_tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_token_observed,
                return_token_type_ids=False,
            ).to(DEVICE_1)
            
            outputs = self.scoring_model(**tokenized_score)
            labels = tokenized_score.input_ids[:, 1:]
            logits_score = outputs.logits[:, :-1]

            # -------------------------------------------------------------------------
            # 新增逻辑：基于 Attention Entropy 动态选择 Layer 并计算权重
            # -------------------------------------------------------------------------
            if outputs.attentions is None:
                raise ValueError("Model did not return attentions.")

            import torch.nn.functional as F

            # 1. 预计算所有层的 Token Entropy
            all_layers_entropy = []
            
            for layer_attn in outputs.attentions:
                # layer_attn: (Batch, Heads, Seq, Seq)
                # 计算每个 Head 的 Entropy
                head_entropy = -torch.sum(layer_attn * torch.log(layer_attn + 1e-10), dim=-1)
                # 对 Heads 取平均: (Batch, Seq)
                token_entropy = head_entropy.mean(dim=1)
                all_layers_entropy.append(token_entropy)            


            # 4. 获取选中层的 Entropy
            # 选取那些层id大于中间位置且entropy小于所有层平均值的那些层，并且对这些层的entropy取平均作为最终的token权重
            attn_mask = tokenized_score.attention_mask.to(all_layers_entropy[0].device).float()
            layer_means = []
            for token_entropy in all_layers_entropy:
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


            # 5. 对齐 Labels (去掉第一个位置): (Batch, Seq-1)
            entropy_slice = selected_token_entropy[:, 1:] 
            
            # 6. 计算权重
            temperature = 0.7
            score_for_softmax = -entropy_slice / temperature

            # 7. 处理 Padding Mask
            valid_mask = tokenized_score.attention_mask[:, 1:].to(DEVICE_1)
            score_for_softmax = score_for_softmax.masked_fill(valid_mask == 0, float('-inf'))
            
            # 8. Softmax 归一化
            token_weights = torch.softmax(score_for_softmax, dim=-1)
            # -------------------------------------------------------------------------
            
            if self.scoring_model_name == self.reference_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_token_observed,
                    return_token_type_ids=False,
                ).to(DEVICE_2)
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]

                min_seq_len = min(logits_ref.shape[1], logits_score.shape[1])
                logits_ref = logits_ref[:, :min_seq_len, :]
                logits_score = logits_score[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len]
                token_weights = token_weights[:, :min_seq_len]
                token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + 1e-8)

            common_device = DEVICE_1
            logits_score = logits_score.to(common_device)
            logits_ref = logits_ref.to(common_device)
            labels = labels.to(common_device)
            token_weights = token_weights.to(common_device)

            scores = self.discrepancy(logits_ref, logits_score, labels, token_weights=token_weights)

        return scores.tolist()
    
    def forward(self, input_text):
        return self.compute_score(input_text)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = FastDetectGPT(
        scoring_model_name=args.scoring_model_name,
        reference_model_name=args.reference_model_name,
        cache_dir=args.cache_dir,
        discrepancy_analytic=args.discrepancy_analytic,
        max_token_observed=args.max_token_observed,
    )

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    human_scores = []
    llm_scores = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    for idx, item in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating FastDetectGPT on {args.eval_data_path.split('/')[-1]}",
    ):
        human_scores.extend(model(item['human']))
        llm_scores.extend(model(item['llm']))

    fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
    prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
    eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)
    human_score_mean = torch.mean(torch.tensor(human_scores)).item()
    human_score_std = torch.std(torch.tensor(human_scores)).item()
    llm_score_mean = torch.mean(torch.tensor(llm_scores)).item()
    llm_score_std = torch.std(torch.tensor(llm_scores)).item()

    print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}')

    result_dict = {
        'method': 'FastDetectGPT',
        'scoring_model_name': args.scoring_model_name,
        'reference_model_name': args.reference_model_name,
        'discrepancy_analytic': args.discrepancy_analytic,
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
        "human_scores": human_scores,
        "llm_scores": llm_scores
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)