# coding=utf-8
# This file is derived from HuggingFace Transformers project
# (https://github.com/huggingface/transformers).
# Licensed under the Apache License, Version 2.0.
# 
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

# from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState
from .logits_process import LogitsProcessorList


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices indicating to which beam hypothesis each token correspond.
            group_index (`int`, *optional*):
                The index of the group of beams. Used with [`~PreTrainedModel.group_beam_search`].

        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of all
                non-finished beams.
                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            final_beam_scores (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
                The final scores of all non-finished beams.
            final_beam_tokens (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
                The last tokens to be added to the non-finished beam_hypotheses.
            final_beam_indices (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
                The beam indices indicating to which beam the `final_beam_tokens` shall be added.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early
            due to the `eos_token_id`.

        """
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        num_actions: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        gym_env_id: Optional[str] = "SmartDryerEnv-v1",
        **kwargs,
    ):
        self.num_beams = num_beams
        self.num_actions = num_actions
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        if num_beam_groups > 1:
            raise NotImplementedError("Beam groups > 1 is not yet supported for RLGBS")
        self.group_size = self.num_beams // self.num_beam_groups

        self.oracle_env = gym.make(gym_env_id)
        self.oracle_env.unwrapped.seed(12450) # Not necessary as beam search scorer's env does not initialize by itself

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        """
        Check if all beams have reached EOS (or have been truncated).
        """
        return self._done.all()

    def rollout_new_hypotheses_in_batch(self, input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses) -> tuple[np.ndarray, int]:
        num_alive_hypotheses = filtered_next_tokens.numel() # Total number of hypotheses that are available
        this_task_size = min(self.oracle_env.unwrapped.cpu_count, num_alive_hypotheses - num_cached_hypotheses)
        if num_alive_hypotheses == 0 or num_alive_hypotheses - num_cached_hypotheses == 0:
            return None, 0
        this_task_next_tokens = filtered_next_tokens[0, num_cached_hypotheses:num_cached_hypotheses+this_task_size]
        this_task_next_indices = filtered_next_indices[0, num_cached_hypotheses:num_cached_hypotheses+this_task_size]
        this_task_base_input_ids = input_ids[this_task_next_indices, :]
        extended_input_ids = torch.cat((this_task_base_input_ids, this_task_next_tokens.unsqueeze(dim=-1)), dim=1)
        _, all_rewards, _ = self.oracle_env.unwrapped.redis_cached_batch_rollout(extended_input_ids)
        return all_rewards, this_task_size

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        oracle_env_rstates: np.ndarray,
        rewards_array: torch.FloatTensor,
        this_done_array: torch.BoolTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        # Reset the environment with the supplied initial conditions
        self.oracle_env.reset(options={"rstates": oracle_env_rstates.squeeze().item() if oracle_env_rstates.size == 1 else oracle_env_rstates})
        # add up to the length which the next_scores is calculated on (including decoder prompt)
        cur_len = input_ids.shape[-1] + 1 # TODO old version doesn't have this "+1"
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not batch_size == (input_ids.shape[0] // self.group_size):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        # TODO: may need to pre-fill the next_beam_scores, next_beam_tokens, next_beam_indices with -inf or something
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # use self.oracle_env.unwrapped.redis_cached_batch_rollout to calulcate rewards and done for next tokens
            # First filter out positions in next_indices that are done
            done_indices = torch.nonzero(this_done_array[:, batch_idx])
            if done_indices.numel() == 0:
                filtered_next_tokens = next_tokens[batch_idx].view(1, -1)
                filtered_next_indices = next_indices[batch_idx].view(1, -1)
            else:
                mask = ~next_indices[batch_idx].eq(done_indices).any(dim=0)
                filtered_next_tokens = next_tokens[batch_idx, mask].view(1, -1)
                filtered_next_indices = next_indices[batch_idx, mask].view(1, -1)
            # num_alive_hypotheses = filtered_next_tokens.numel() # Total number of hypotheses that are available
            num_cached_hypotheses = 0 # Total number of hypotheses that have been rolled out and stored in the cache
            hypothesis_rewards, this_task_size = self.rollout_new_hypotheses_in_batch(input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses)
            num_cached_hypotheses += this_task_size
            cache_increment_size = this_task_size # This should be the max. num of hypotheses that can be rolled out in one go

            # Check if input_ids is done in parallel to save time
            # _, rewards_array, this_done_array = self.oracle_env.unwrapped.redis_cached_batch_rollout(input_ids)
            # rewards_array = rewards_array.to(input_ids.device)
            # this_done_array = this_done_array.to(input_ids.device)
            # next tokens for this sentence.
            cached_run_idx = 0
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                # print(beam_token_rank, (next_token, next_score, next_index))
                batch_beam_idx = batch_idx * self.group_size + next_index

                # this_done = self.bsenv.done_before_nextid(input_ids[batch_beam_idx])
                this_done = this_done_array[batch_beam_idx].item()
                this_reward = rewards_array[batch_beam_idx].item() # Higher scores are preferred by beam search hypotheses

                # add to generated hypotheses if end of sentence
                # if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                if this_done and next_score.item() > -float("inf"):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    # is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    # if is_beam_token_worse_than_top_num_beams:
                    #     continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    # self._beam_hyps[batch_group_idx].add(
                    #     input_ids[batch_beam_idx].clone(),
                    #     next_score.item(),
                    #     beam_indices=beam_index,
                    #     generated_len=cur_len - decoder_prompt_len,
                    # )
                    # rate hypotheses using actual reward instead
                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        this_reward, # Using actual reward score instead of beam score for rating hypotheses
                        # next_score.item(), # This actually may contain one extra score, need fix later
                        beam_indices=beam_index,
                        generated_len=cur_len - decoder_prompt_len,
                    )
                    # print(f"[BeamSearchScorer.process] {input_ids[batch_beam_idx]} done and added to hyp")
                    # self._done[batch_idx] = True # Uncomment this line to stop beam search when first beam of length n is done. This ensures no beams of length > n will be explored, and may result in repetitive sequences.
                else:
                    # If we ran out of cached hypotheses, request a new batch
                    if cached_run_idx >= num_cached_hypotheses:
                        hypothesis_rewards, this_task_size = self.rollout_new_hypotheses_in_batch(input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses)
                        num_cached_hypotheses += this_task_size
                    if hypothesis_rewards is None:
                        break
                    # If the reward for this new hypothesis is bad (<0), we should not add it to the beam
                    if hypothesis_rewards[cached_run_idx%cache_increment_size, batch_idx] >= 0:
                        # add next predicted token since it is not eos_token
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1
                    cached_run_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            # TODO re-enable this later
            # if beam_idx < self.group_size:
            #     raise ValueError(
            #         f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
            #         f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
            #     )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def _refine_best_hypothesis(
        self, 
        input_hypothesis: torch.LongTensor, 
        input_score: float, 
        logits_processor: Optional[LogitsProcessorList] = None,
        m: Optional[int] = 2,
    ) -> tuple[torch.LongTensor, float]:
        # Get number of possible actions
        n_actions = self.oracle_env.unwrapped.action_space.n
        input_device = input_hypothesis.device
        best_trajectory = input_hypothesis
        best_score = input_score
        
        for append_length in range(1, m+1):
            # Create all possible combinations for last m positions
            seed_trajectory = input_hypothesis[:-m].unsqueeze(0) # We always keep the first (L - m) actions regardless of the append length
            
            # Create cartesian product for last m positions
            combinations = torch.cartesian_prod(*[torch.arange(n_actions) for _ in range(append_length)]).to(input_device)
            if append_length == 1:
                combinations = combinations.view(-1, 1)
            
            # Repeat seed trajectory for each combination
            seed_trajectory = seed_trajectory.repeat_interleave(len(combinations), dim=0)
            
            # Combine seed with all possible combinations
            extended_input_ids = torch.cat((seed_trajectory, combinations), dim=1)
            
            # Process logits constraints if provided
            if logits_processor is not None:
                valid_sequences = []
                for seq in extended_input_ids:
                    is_valid = True
                    for i in range(len(seq) - m, len(seq)):
                        dummy_logits = torch.ones((1, n_actions)).to(input_device) / n_actions
                        processed_logits = logits_processor(seq[:i].unsqueeze(0), dummy_logits)
                        if processed_logits[0, seq[i].item()] == -float("inf"):
                            is_valid = False
                            break
                    if is_valid:
                        valid_sequences.append(seq)
                extended_input_ids = torch.stack(valid_sequences) if valid_sequences else extended_input_ids[:0]
            
            # Skip to next length if no valid sequences found
            if len(extended_input_ids) == 0:
                continue
            
            # Evaluate all valid combinations
            _, all_rewards, all_dones = self.oracle_env.unwrapped.redis_cached_batch_rollout(extended_input_ids)
            
            # Find best performing sequence that has done == True
            # Set all_rewards to -inf for all sequences that have done == False
            all_rewards[all_dones == False] = -float("inf")
            best_idx = torch.argmax(all_rewards).item()
            this_best_score = all_rewards[best_idx].item()
            this_best_trajectory = extended_input_ids[best_idx]

            # Update best trajectory and score if this one is better
            if this_best_score > best_score:
                best_score = this_best_score
                best_trajectory = this_best_trajectory

        if best_score > input_score:
            print(f"[BeamSearchScorer._refine_best_hypothesis] Refined reward: [{best_score:.4f}], original reward: [{input_score:.4f}]")
        
        if input_hypothesis.numel() != best_trajectory.numel():
            # We will need to pad the best trajectory to the same length as the input hypothesis
            # The shape of trajectory is (seq_len,) and padding token is -2 (hardcoded for now)
            best_trajectory = torch.cat((best_trajectory, torch.tensor([-2 for _ in range(input_hypothesis.numel() - best_trajectory.numel())]).to(input_device)), dim=0)
        
        return best_trajectory, best_score


    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
        max_refine_steps: Optional[int] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # We shouldn't need to do this as any completed beams should have already been added to the beam hypothesis
        # Which means that the rest of the beams are either not completed or are not valid
        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            beams_collect = []
            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                generated_len = final_tokens.shape[-1] - decoder_prompt_len
                # Instead of adding the beam to the beam hypothesis, we add it to the list of beams to be considered
                # beam_hyp.add(final_tokens, final_score, beam_indices=beam_index, generated_len=generated_len)
                if final_score > -float("inf"):
                    beams_collect.append((final_tokens, beam_index, generated_len))

            # Evaluate the beams that were to be added to the beam hypothesis # TODO this might not work if lentghs are different
            if len(beams_collect) > 0:
                final_tokens_concatenated = torch.cat([beam[0].unsqueeze(0) for beam in beams_collect], dim=0)
                # Evaluate the rewards of the beams using the oracle environment
                _, all_rewards, all_dones = self.oracle_env.unwrapped.redis_cached_batch_rollout(final_tokens_concatenated)
                # Add the beams to the beam hypothesis using the rewards instead of the beam scores
                for i, beam in enumerate(beams_collect):
                    # Only add if the beam is done
                    if all_dones[i].item():
                        final_tokens = beam[0]
                        final_reward = all_rewards[i].item()
                        beam_index = beam[1]
                        generated_len = beam[2]
                        beam_hyp.add(final_tokens, final_reward, beam_indices=beam_index, generated_len=generated_len)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                # best_hyp_tuple = sorted_hyps.pop()
                best_hyp_tuple = sorted_hyps[max(0, len(sorted_hyps)-1-j)] # NOTE uncomment this to allow for repetitive hyps
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # Attempt to refine the best hypothesis by replacing the last token with every possible token
                # and selecting the one with the highest reward
                if max_refine_steps > 0:
                    refined_best_hyp, refined_best_score = self._refine_best_hypothesis(best_hyp, best_score, logits_processor, max_refine_steps)
                else:
                    refined_best_hyp = best_hyp
                    refined_best_score = best_score

                # append hyp to lists
                best.append(refined_best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = refined_best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformers.BeamSearchScorer.finalize`].
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        num_actions: int,
        constraints: List[Constraint],
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        gym_env_id: Optional[str] = "SmartDryerEnv-v1",
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        if num_beam_groups > 1:
            raise NotImplementedError("Beam groups > 1 is not yet supported for RLCBS")
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        self.oracle_env = gym.make(gym_env_id)
        self.oracle_env.unwrapped.seed(12450) # Not necessary as beam search scorer's env does not initialize by itself

        self._is_init = False
        # HF uses num_beams here instead of group_size? Does this suggest that CBS doesn't support grouped BS at all?
        # https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/generation/beam_search.py#L480
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def make_constraint_states(self, n):
        return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        new_state = self.make_constraint_states(1)[0]
        new_state.reset(sequence)
        return new_state.completed

    def rollout_new_hypotheses_in_batch(self, input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses) -> tuple[np.ndarray, int]:
        num_alive_hypotheses = filtered_next_tokens.numel() # Total number of hypotheses that are available
        this_task_size = min(self.oracle_env.unwrapped.cpu_count, num_alive_hypotheses - num_cached_hypotheses)
        if num_alive_hypotheses == 0 or num_alive_hypotheses - num_cached_hypotheses == 0:
            return None, 0
        this_task_next_tokens = filtered_next_tokens[0, num_cached_hypotheses:num_cached_hypotheses+this_task_size]
        this_task_next_indices = filtered_next_indices[0, num_cached_hypotheses:num_cached_hypotheses+this_task_size]
        this_task_base_input_ids = input_ids[this_task_next_indices, :]
        extended_input_ids = torch.cat((this_task_base_input_ids, this_task_next_tokens.unsqueeze(dim=-1)), dim=1)
        _, all_rewards, _ = self.oracle_env.unwrapped.redis_cached_batch_rollout(extended_input_ids)
        return all_rewards, this_task_size

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        oracle_env_rstates: np.ndarray,
        rewards_array: torch.FloatTensor,
        this_done_array: torch.BoolTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices indicating to which beam hypothesis each token correspond.
            decoder_prompt_len (`int`, *optional*):
                The length of prompt that is included in the input to decoder.
        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.

                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """

        # Reset the environment with the supplied initial conditions
        self.oracle_env.reset(options={"rstates": oracle_env_rstates.squeeze().item() if oracle_env_rstates.size == 1 else oracle_env_rstates})
        # add up to the length which the next_scores is calculated on (including decoder prompt)
        cur_len = input_ids.shape[-1] + 1 # NOTE old versions of HF transformer didn't have this +1
        batch_size = len(self._beam_hyps)
        if not batch_size == (input_ids.shape[0] // self.group_size):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # use self.oracle_env.unwrapped.redis_cached_batch_rollout to calulcate rewards and done for next tokens
            # First filter out positions in next_indices that are done
            done_indices = torch.nonzero(this_done_array[:, batch_idx])
            if done_indices.numel() == 0:
                filtered_next_tokens = next_tokens[batch_idx].view(1, -1)
                filtered_next_indices = next_indices[batch_idx].view(1, -1)
            else:
                mask = ~next_indices[batch_idx].eq(done_indices).any(dim=0)
                filtered_next_tokens = next_tokens[batch_idx, mask].view(1, -1)
                filtered_next_indices = next_indices[batch_idx, mask].view(1, -1)
            # num_alive_hypotheses = filtered_next_tokens.numel() # Total number of hypotheses that are available
            num_cached_hypotheses = 0 # Total number of hypotheses that have been rolled out and stored in the cache
            hypothesis_rewards, this_task_size = self.rollout_new_hypotheses_in_batch(input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses)
            num_cached_hypotheses += this_task_size
            cache_increment_size = this_task_size # This should be the max. num of hypotheses that can be rolled out in one go

            # Check if input_ids is done in parallel to save time
            # _, rewards_array, this_done_array = self.oracle_env.unwrapped.redis_cached_batch_rollout(input_ids)
            # rewards_array = rewards_array.to(input_ids.device)
            # this_done_array = this_done_array.to(input_ids.device)
            # next tokens for this sentence.
            cached_run_idx = 0
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index

                # this_done = self.bsenv.done_before_nextid(input_ids[batch_beam_idx])
                this_done = this_done_array[batch_beam_idx].item()
                this_reward = rewards_array[batch_beam_idx].item() # Higher scores are preferred by beam search hypotheses

                # add to generated hypotheses if end of sentence
                # if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                if this_done and next_score.item() > -float("inf"):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    # is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    # if is_beam_token_worse_than_top_num_beams:
                    #     continue
                    # TODO one could possibly refine the done trajectory by swapping the last id with
                    # every id possible and evaluate them, but this could be very expensive
                    completes_constraint = self.check_completes_constraints(input_ids[batch_beam_idx].cpu().tolist())
                    if completes_constraint:
                        if beam_indices is not None:
                            beam_index = beam_indices[batch_beam_idx]
                            beam_index = beam_index + (batch_beam_idx,)
                        else:
                            beam_index = None

                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            this_reward, # Using actual reward score instead of beam score for rating hypotheses
                            # next_score.item(),
                            beam_indices=beam_index,
                            generated_len=cur_len - decoder_prompt_len,
                        )
                        # print(f"[ConstrainedBeamSearchScorer.process] {input_ids[batch_beam_idx]} done and added to hyp")
                        # self._done[batch_idx] = True # Uncomment this line to stop beam search when first beam of length n is done. This ensures no beams of length > n will be explored, and may result in repetitive sequences.
                    # else:
                        # print(f"[ConstrainedBeamSearchScorer.process] {input_ids[batch_beam_idx]} done but did not satisfy all constraints")
                else:
                    # If we ran out of cached hypotheses, request a new batch
                    if cached_run_idx >= num_cached_hypotheses:
                        hypothesis_rewards, this_task_size = self.rollout_new_hypotheses_in_batch(input_ids, filtered_next_tokens, filtered_next_indices, num_cached_hypotheses)
                        num_cached_hypotheses += this_task_size
                    if hypothesis_rewards is None:
                        break
                    # If the reward for this new hypothesis is bad (<0), we should not add it to the beam
                    if hypothesis_rewards[cached_run_idx%cache_increment_size, batch_idx] >= 0: # NOTE this threshold could be worth tuning
                        # add next predicted token since it is not eos_token
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1
                    cached_run_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            new_scores, new_tokens, new_indices = self.step_sentence_constraint(
                batch_idx,
                input_ids,
                scores_for_all_vocab,
                next_beam_scores[batch_idx],
                next_beam_tokens[batch_idx],
                next_beam_indices[batch_idx],
            )

            next_beam_scores[batch_idx] = new_scores
            next_beam_tokens[batch_idx] = new_tokens
            next_beam_indices[batch_idx] = new_indices

            # TODO re-enable this later
            # if beam_idx < self.group_size:
            #     print(f"Suppressed error: At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`")
            #     raise ValueError(
            #         f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
            #         f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
            #     )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = False,
    ):
        # sent_beam_tokens are the next {num_beams} number of tokens that are under consideration for this beam
        # (candidate next tokens)

        # 1. Adding "advance_tokens"
        #     using ConstraintStateList.advance(), we propose new tokens to be added into this "candidate list" that will
        #     advance us in fulfilling the constraints.

        # 2. Selecting best candidates such that we end up with highest probable candidates
        #     that fulfill our constraints.

        orig_len = sent_beam_indices.size(0)
        device = sent_beam_indices.device

        # initialize states
        topk_contraint_states = self.make_constraint_states(orig_len)
        advance_constraint_states = self.make_constraint_states(orig_len)

        sidx, eidx = batch_idx * orig_len, (batch_idx + 1) * orig_len
        this_batch_input_ids = input_ids[sidx:eidx]
        this_batch_token_scores = vocab_scores[sidx:eidx]
        full_hypotheses = torch.cat((input_ids[sent_beam_indices], sent_beam_tokens.unsqueeze(-1)), dim=-1)

        # need to make new hypothesis that advance the constraints
        track_new = {
            "new_seqs": full_hypotheses.tolist(),
            "new_states": [],
            "new_indices": [],
            "new_tokens": [],
            "new_scores": [],
        }
        for seq_idx, pre_seq in enumerate(this_batch_input_ids):
            # pre_seq = ith sequence generated before this step.

            # input_ids -> (topk) generic beam search best model next tokens
            #           -> (advance) constraints forcing the next token
            # either way, we need to sort them into "banks" later, so store a "ConstraintListState" for all types of
            # hypotheses.

            topk_state = topk_contraint_states[seq_idx]
            topk_state.reset(full_hypotheses[seq_idx].cpu().tolist())

            advance_state = advance_constraint_states[seq_idx]
            advance_state.reset(pre_seq.cpu().tolist())

            if not advance_state.completed:
                advance_tokens = torch.LongTensor(advance_state.advance()).to(device)
                for advance_token in advance_tokens:
                    # since adding each `advance_token` leads to a different hypothesis, create new state instance.
                    new_state = advance_state.copy(stateful=True)
                    new_state.add(advance_token.cpu().tolist())

                    advance_seq = torch.cat((pre_seq, advance_token.unsqueeze(0)), -1).cpu().tolist()
                    if advance_seq not in track_new["new_seqs"]:
                        # prevent duplicates, which are basically bound to happen in this process.
                        track_new["new_seqs"].append(advance_seq)
                        track_new["new_indices"].append(sidx + seq_idx)  # idx -> global idx across all the batches
                        track_new["new_tokens"].append(advance_token)
                        track_new["new_scores"].append(this_batch_token_scores[seq_idx].take(advance_token))
                        track_new["new_states"].append(new_state)
            elif push_progress:
                # Basically, `sent_beam_indices` often chooses very little among `input_ids` the generated sequences that
                # actually fulfill our constraints. For example, let constraints == ["loves pies"] and

                #     pre_seq_1 = "The child loves pies and" pre_seq_2 = "The child plays in the playground and"

                # Without this step, if `sent_beam_indices` is something like [1,1], then
                #     1. `pre_seq_1` won't be added to the list of (topk) hypothesis since it's not in the indices and
                #     2.  it won't be added to the list of (advance) hypothesis since it's completed already. (this is
                #         the else part of `if constraints_completed[seq_idx]`)
                #     3. it ends up simply getting removed from consideration.

                # #3 might be fine and actually desired, since it's likely that it's a low-probability output anyways,
                # especially if it's not in the list of `sent_beam_indices`. But this often leads to lengthened beam
                # search times, since completed sequences keep getting removed after all this effort for constrained
                # generation.

                # Here, we basically take `pre_seq_1` and to "push" it into the considered list of hypotheses, by simply
                # appending the next likely token in the vocabulary and adding it to the list of hypotheses.

                new_score, new_token = torch.max(this_batch_token_scores[seq_idx], 0)  # some next probable token
                advance_seq = torch.cat((pre_seq, new_token.unsqueeze(0)), -1)

                advance_state = advance_constraint_states[seq_idx]

                advance_seq = advance_seq.cpu().tolist()

                advance_state.reset(advance_seq)
                if advance_seq not in track_new["new_seqs"]:
                    # but still don't want to have duplicates
                    track_new["new_seqs"].append(advance_seq)
                    track_new["new_indices"].append(seq_idx)
                    track_new["new_tokens"].append(new_token)
                    track_new["new_scores"].append(new_score)
                    track_new["new_states"].append(advance_state)

        if len(track_new["new_indices"]) > 0:
            new_indices = torch.tensor(track_new["new_indices"]).to(device)
            new_tokens = torch.stack(track_new["new_tokens"]).to(device)
            new_scores = torch.stack(track_new["new_scores"]).to(device)

            all_states = topk_contraint_states + track_new["new_states"]
            all_tokens = torch.cat((sent_beam_tokens, new_tokens), -1)
            all_scores = torch.cat((sent_beam_scores, new_scores), -1)
            all_banks = torch.tensor([one.get_bank() for one in all_states]).to(device)

            zipped = all_banks * 100 + all_scores
            indices = zipped.sort(descending=True).indices
            sorted_banks = all_banks[indices]

            # Then we end up with {sorted among bank C}, {sorted among bank C-1}, ..., {sorted among bank 0}

            counter = -1
            cur_bank = sorted_banks[0]
            increments = []
            for bank in sorted_banks:
                if bank == cur_bank:
                    counter += 1
                else:
                    counter = 0
                    cur_bank = bank
                increments.append(counter)
            rearrangers = torch.tensor(np.argsort(increments, kind="mergesort"))

            indices = indices[rearrangers][:orig_len]

            sent_beam_scores = all_scores[indices]
            sent_beam_tokens = all_tokens[indices]
            sent_beam_indices = torch.cat((sent_beam_indices, new_indices))[indices]

        return sent_beam_scores, sent_beam_tokens, sent_beam_indices

    def _refine_best_hypothesis(self, input_hypothesis: torch.LongTensor, input_score: float, logits_processor: Optional[LogitsProcessorList] = None) -> torch.LongTensor:
        # There are total self.oracle_env.unwrapped.action_space.n possible tokens
        n_actions = self.oracle_env.unwrapped.action_space.n
        seed_trajectory = input_hypothesis[:-1].unsqueeze(0).repeat_interleave(n_actions, dim=0)
        action_postfix = torch.arange(0, n_actions).view(-1, 1).to(input_hypothesis.device)
        # original_tair_level = input_hypothesis[-1].item() % 11
        extended_input_ids = torch.cat((seed_trajectory, action_postfix), dim=1)
        # For each row in extended_input_ids, check whether they satisfied the constraints
        # negative constraints
        dummy_next_token_logits = torch.ones((1, n_actions)).to(input_hypothesis.device) / n_actions
        processed_dummy_logits = logits_processor(input_hypothesis[:-1].unsqueeze(0), dummy_next_token_logits)
        negative_constraints_mask = (processed_dummy_logits > -float("inf")).squeeze(0)
        # positive constraints
        positive_constraints_mask = torch.tensor([self.check_completes_constraints(extended_input_ids[i].cpu().tolist()) for i in range(n_actions)], dtype=torch.bool).to(input_hypothesis.device)
        # Pass only the rows that satisfy the constraints to the oracle environment
        mask = torch.logical_and(positive_constraints_mask, negative_constraints_mask)
        extended_input_ids = extended_input_ids[mask]
        if extended_input_ids.shape[0] <= 1:
            # print("[ConstrainedBeamSearchScorer._refine_best_hypothesis] Failed to find new hypothesis satisfying all constraints")
            return input_hypothesis, input_score
        _, all_rewards, _ = self.oracle_env.unwrapped.redis_cached_batch_rollout(extended_input_ids)
        # Find the index of highest reward
        best_idx = np.argmax(all_rewards)
        best_score = all_rewards[best_idx]
        best_trajectory = extended_input_ids[best_idx, :]
        if best_score.item() > input_score:
            print(f"[ConstrainedBeamSearchScorer._refine_best_hypothesis] Refined reward: [{best_score.item():.4f}], original reward: [{input_score:.4f}]")
        return best_trajectory, best_score.item()

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
        max_refine_steps: Optional[int] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # We shouldn't need to do this as any completed beams should have already been added to the beam hypothesis
        # Which means that the rest of the beams are either not completed or are not valid
        # # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams

            ids_collect = []
            beams_collect = []
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item() # TODO these scores seem like a mix of real rewards and sum log probabilities
                final_tokens = input_ids[batch_beam_idx]

                completes_constraint = self.check_completes_constraints(final_tokens.cpu().tolist())
                if completes_constraint:
                    beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                    generated_len = final_tokens.shape[-1] - decoder_prompt_len
                    # Instead of adding the beam to the beam hypothesis, we add it to the list of beams to be considered
                    # beam_hyp.add(final_tokens, final_score, beam_indices=beam_index, generated_len=generated_len)
                    if final_score > float("-inf"):
                        beams_collect.append((final_tokens, beam_index, generated_len))
                        ids_collect.append(beam_id)

            # Evaluate the beams that were to be added to the beam hypothesis # TODO this might not work if lentghs are different
            if len(beams_collect) > 0:
                final_tokens_concatenated = torch.cat([beam[0].unsqueeze(0) for beam in beams_collect], dim=0)
                # Evaluate the rewards of the beams using the oracle environment
                _, all_rewards, all_dones = self.oracle_env.unwrapped.redis_cached_batch_rollout(final_tokens_concatenated)
                # Add the beams to the beam hypothesis using the rewards instead of the beam scores
                for i, beam in enumerate(beams_collect):
                    # Only add if the beam is done
                    if all_dones[i].item():
                        final_tokens = beam[0]
                        final_reward = all_rewards[i].item()
                        beam_index = beam[1]
                        generated_len = beam[2]
                        beam_hyp.add(final_tokens, final_reward, beam_indices=beam_index, generated_len=generated_len)

            # due to overly complex constraints or other factors, sometimes we can't guarantee a successful
            # generation. In these cases we simply return the highest scoring outputs.
            # if len(ids_collect) < self.num_beam_hyps_to_keep:
            #     for beam_id in range(self.num_beams):
            #         if beam_id not in ids_collect:
            #             batch_beam_idx = batch_idx * self.num_beams + beam_id
            #             final_score = final_beam_scores[batch_beam_idx].item()
            #             final_tokens = input_ids[batch_beam_idx]
            #             generated_len = final_tokens.shape[-1] - decoder_prompt_len
            #             beam_hyp.add(final_tokens, final_score, generated_len=generated_len)
            #         if len(ids_collect) >= self.num_beam_hyps_to_keep:
            #             break

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
        # TODO: This could explain why sometimes the beam search doesn't return the best results - all existing hypotheses are negative and making the dummy 0 hypothesis the best

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                # best_hyp_tuple = sorted_hyps.pop()
                best_hyp_tuple = sorted_hyps[max(0, len(sorted_hyps)-1-j)] # NOTE uncomment this to allow for repetitive hyps
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # Attempt to refine the best hypothesis by replacing the last token with every possible token
                # and selecting the one with the highest reward
                if max_refine_steps > 0:
                    # Run the refinement process but dont actually replace the best hypothesis
                    # TODO: support for multiple refinement steps
                    refined_best_hyp, refined_best_score = self._refine_best_hypothesis(best_hyp, best_score, logits_processor)
                else:
                    refined_best_hyp = best_hyp
                    refined_best_score = best_score

                # append hyp to lists
                best.append(refined_best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = refined_best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1

        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self,
            hyp: torch.LongTensor,
            score: float,
            beam_indices: Optional[torch.LongTensor] = None,
            generated_len: Optional[int] = None):
        """
        Add a new hypothesis to the list, without applying any length penalties to the score.
        """
        if len(self) < self.num_beams or score > self.worst_score:
            # Check if hyp is already in the list, only add if it is not. This prevents duplicate hypotheses in returned sequences
            for _, existing_hyp, _ in self.beams:
                if torch.equal(hyp, existing_hyp):
                    return
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    # def add(
    #     self,
    #     hyp: torch.LongTensor,
    #     sum_logprobs: float,
    #     beam_indices: Optional[torch.LongTensor] = None,
    #     generated_len: Optional[int] = None,
    # ):
    #     """
    #     Add a new hypothesis to the list.
    #     """
    #     if generated_len is not None:
    #         score = sum_logprobs / (generated_len**self.length_penalty)
    #     # This 'else' case exists for retrocompatibility
    #     else:
    #         score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

    #     if len(self) < self.num_beams or score > self.worst_score:
    #         self.beams.append((score, hyp, beam_indices))
    #         if len(self) > self.num_beams:
    #             sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
    #             del self.beams[sorted_next_scores[0][1]]
    #             self.worst_score = sorted_next_scores[1][0]
    #         else:
    #             self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = 0) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        # elif self.early_stopping is False:
        #     highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
        #     ret = self.worst_score >= highest_attainable_score
        #     return ret
        # # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        # else:
        #     # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
        #     # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
        #     # its max this way
        #     if self.length_penalty > 0.0:
        #         if self.max_length <= decoder_prompt_len:
        #             raise ValueError("max_length is not larger than decoder prompt length")
        #         highest_attainable_score = (
        #             best_sum_logprobs / (self.max_length - decoder_prompt_len) ** self.length_penalty
        #         )
        #     # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
        #     else:
        #         highest_attainable_score = best_sum_logprobs / (cur_len - decoder_prompt_len) ** self.length_penalty
        #     ret = self.worst_score >= highest_attainable_score
        #     return ret
        if cur_len + 1 >= 14:
            return True
        # cur_score = best_sum_logprobs / cur_len**self.length_penalty
        # ret = self.worst_score >= cur_score
        # return ret
        return False
