# Part of codes in this file was copied from project [vLLM Team][vllm]
import random
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from vllm.sampling_params import SamplingType
from vllm.sequence import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.model_executor.layers.sampler import _get_logprobs, _build_sampler_output
from mindie_llm.text_generator.utils.sampling_metadata import SamplingData, SamplingParam

_SAMPLING_EPS = 1e-5
SampleResultType = List[Tuple[List[int], List[int]]]


def _to_tensor(data, dtype=None):
    if dtype:
        return torch.tensor(data, dtype=dtype, device=torch.device("npu"))
    else:
        return torch.tensor(data, device=torch.device("npu"))


class AscendSampler(nn.Module):
    def __init__(self, mindie_model):
        super().__init__()
        self.mindie_model = mindie_model
        self.include_gpu_probs_tensor = False

    def forward(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        _, vocab_size = logits.shape
        mindie_sampling_data, mindie_sampling_param = self.construct_data(sampling_metadata, vocab_size)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        next_tokens = self.mindie_model.sample(
            logits,
            sampling_data=mindie_sampling_data,
            sampling_param=mindie_sampling_param,
        )

        sample_results, maybe_sampled_tokens_tensor = recover_data(
            sampling_metadata=sampling_metadata,
            sampled_tokens=next_tokens,
            logprobs=logprobs,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
        )
        if self.include_gpu_probs_tensor:
            if maybe_sampled_tokens_tensor is None:
                raise RuntimeError("maybe_sampled_tokens_tensor is None")
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            on_device_tensors = None
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results,
                                     sampling_metadata,
                                     prompt_logprobs,
                                     sample_logprobs,
                                     on_device_tensors=on_device_tensors)

    def construct_data(
            self,
            sampling_metadata: SamplingMetadata,
            vocab_size: int,
    ) -> Tuple[SamplingData, SamplingParam]:
        all_input_tokens: List[List[int]] = []
        prompt_tokens: List[List[int]] = []
        output_tokens: List[List[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        sampling_seeds: List[int] = []
        sample_indices: List[int] = []
        do_samples: List[bool] = []  # To Do
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False
        greedy_flag = False

        if sampling_metadata.seq_groups is None:
            raise RuntimeError("sampling_metadata.seq_group is None, no data received.")
        for seq_group in sampling_metadata.seq_groups:
            do_samples.append(seq_group.do_sample)
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p
            is_greedy = sampling_params.sampling_type == SamplingType.GREEDY
            seed = sampling_params.seed
            if seed is None:
                if is_greedy:
                    seed = 0
                else:
                    lo, hi = torch.iinfo(torch.long).min, torch.iinfo(torch.long).max
                    seed = random.randint(lo, hi)
            if is_greedy:
                greedy_flag = True
            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                temperature = 1.0
            if not do_top_p_top_k and (top_p < 1.0 - _SAMPLING_EPS
                                       or top_k != vocab_size):
                do_top_p_top_k = True
            if not do_min_p and min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties:
                if abs(p) >= _SAMPLING_EPS:
                    do_penalties = True
                elif abs(f) >= _SAMPLING_EPS:
                    do_penalties = True
                elif abs(r - 1.0) >= _SAMPLING_EPS:
                    do_penalties = True
            is_prompt = seq_group.is_prompt
            if (seq_group.is_prompt
                    and sampling_params.prompt_logprobs is not None):
                # For tokens in the prompt that we only need to get
                # their logprobs
                query_len = seq_group.query_len
                if query_len is None:
                    raise RuntimeError("query_len is None")
                prefill_len = len(seq_group.prompt_logprob_indices)
                temperatures += [temperature] * prefill_len
                sampling_seeds += [seed] * prefill_len
                top_ps += [top_p] * prefill_len
                top_ks += [top_k] * prefill_len
                min_ps += [min_p] * prefill_len
                presence_penalties += [0] * prefill_len
                frequency_penalties += [0] * prefill_len
                repetition_penalties += [1] * prefill_len
                prompt_tokens.extend([] for _ in range(prefill_len))
                output_tokens.extend([] for _ in range(prefill_len))
                all_input_tokens.extend([] for _ in range(prefill_len))
            if seq_group.do_sample:
                sample_lens = len(seq_group.sample_indices)
                if sample_lens != len(seq_ids):
                    raise ValueError("sample_lens != len(seq_ids)")
                for seq_id in seq_ids:
                    seq_data = seq_group.seq_data[seq_id]
                    prompt_tokens.append(seq_data.prompt_token_ids)
                    output_tokens.append(seq_data.output_token_ids)
                    all_input_tokens.append(seq_data.prompt_token_ids + seq_data.output_token_ids)
                temperatures += [temperature] * len(seq_ids)
                sampling_seeds += [seed] * len(seq_ids)
                top_ps += [top_p] * len(seq_ids)
                top_ks += [top_k] * len(seq_ids)
                min_ps += [min_p] * len(seq_ids)
                presence_penalties += [p] * len(seq_ids)
                frequency_penalties += [f] * len(seq_ids)
                repetition_penalties += [r] * len(seq_ids)
        repetition_penalties = np.array(repetition_penalties, dtype=np.float32)
        frequency_penalties = np.array(frequency_penalties, dtype=np.float32)
        presence_penalties = np.array(presence_penalties, dtype=np.float32)
        temperatures = np.array(temperatures, dtype=np.float32)
        top_ks = np.array(top_ks, dtype=np.int32)
        top_ps = np.array(top_ps, dtype=np.float32)
        sampling_seeds = np.array(sampling_seeds)
        do_samples = np.array(do_samples)
        max_tokens_len = max([len(tokens) for tokens in all_input_tokens], default=0)
        padded_all_input_tokens = [
            tokens + [vocab_size] * (max_tokens_len - len(tokens))
            for tokens in all_input_tokens
        ]
        padded_all_input_tokens = np.array(padded_all_input_tokens, dtype=np.int32)
        output_max_len = max([len(tokens) for tokens in output_tokens], default=0)
        padded_output_tokens = [
            tokens + [vocab_size] * (output_max_len - len(tokens))
            for tokens in output_tokens
        ]
        padded_output_tokens = np.array(padded_output_tokens, dtype=np.int32)
        all_input_ids_tensor = _to_tensor(
            padded_all_input_tokens,
            torch.int32
        ) if padded_all_input_tokens is not None else None
        output_ids_tensor = _to_tensor(
            padded_output_tokens,
            torch.int32
        ) if padded_output_tokens is not None else None
        mindie_sampling_data = SamplingData(
            all_input_ids=all_input_ids_tensor,
            output_ids=output_ids_tensor
        )
        if greedy_flag:
            mindie_sampling_param = None
        else:
            mindie_sampling_param = SamplingParam.from_numpy(
                repetition_penalty=repetition_penalties,
                frequency_penalty=frequency_penalties,
                presence_penalty=presence_penalties,
                temperature=temperatures,
                top_k=top_ks,
                top_p=top_ps,
                seed=sampling_seeds,
                do_sample=do_samples,
                to_tensor=_to_tensor,
            )
        return (mindie_sampling_data, mindie_sampling_param)


def recover_data(
        sampling_metadata: SamplingMetadata,
        sampled_tokens: np.ndarray,
        logprobs: torch.Tensor,
        include_gpu_probs_tensor: bool,
) -> Tuple[SampleResultType, Optional[torch.Tensor]]:
    categorized_seq_group_ids: Dict[SamplingType,
    List[int]] = {t: []
                  for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)
    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(logprobs.shape[0],
                                               1,
                                               dtype=torch.long,
                                               device=logprobs.device)
    else:
        sampled_token_ids_tensor = None
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type][:, 0]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)
    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_id, seq_groups) = sample_metadata[sampling_type]
        if sampling_type in (SamplingType.GREEDY, SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = normal_wrap(seq_groups, sampled_tokens)
        elif sampling_type == SamplingType.BEAM:
            sample_results = beam_wrap(seq_groups, sampled_tokens)
        sample_results_dict.update(zip(seq_group_id, sample_results))
    sample_results = [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results, sampled_token_ids_tensor


def normal_wrap(
        selected_seq_groups: List[SequenceGroupToSample],
        samples: np.ndarray,
):
    samples = samples.tolist()
    sample_idx = 0
    results: SampleResultType = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue
        seq_ids = seq_group.seq_ids
        num_parent_seqs = len(seq_ids)
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def beam_wrap(
        selected_seq_groups: List[SequenceGroupToSample],
        samples: np.ndarray,
):
    raise ValueError(f"Unsupported sampling type: beam search")