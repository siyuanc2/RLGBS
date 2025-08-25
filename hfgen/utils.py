# coding=utf-8
# This file is derived from HuggingFace Transformers project
# (https://github.com/huggingface/transformers).
# Licensed under the Apache License, Version 2.0.
# 
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from definitions import *
from stable_baselines3.common.distributions import MultiCategoricalDistribution

from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import (BeamScorer, BeamSearchScorer,
                          ConstrainedBeamSearchScorer)
from .configuration_utils import GenerationConfig, GenerationMode
from .hf_logging import get_logger
from .logits_process import (LogitNormalization, LogitsProcessorList,
                             PrefixConstrainedLogitsProcessor,
                             RLCBSDiscardFailedSequencesLogitsProcessor,
                             RLCBSMaxNTokensLogitsProcessor,
                             SuppressTokensLogitsProcessor)
from .sb3_normalizer import AutoNormalizer
from .stopping_criteria import (EosTokenCriteria, MaxLengthCriteria,
                                MaxTimeCriteria, StoppingCriteria,
                                StoppingCriteriaList,
                                validate_stopping_criteria)

logger = get_logger(__name__)


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and
          `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if `num_beams>1`
          and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if `num_beams>1`
          and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if
          `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if
            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        init_rstates: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function tries to create the initial input_ids from given information.
        """
        input_name = None
        batch_size = 1
        if not inputs:
            # Assume batch_size = 1
            inputs = torch.ones((batch_size, 1), dtype=torch.long, device=self.device, requires_grad=False) * bos_token_id

        return inputs, input_name, model_kwargs

    def _prepare_generation_config(
        self, generation_config: GenerationConfig, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # TODO: provide default GenerationConfig parameters for RL-CBS
            generation_config = GenerationConfig(
                num_beams=4, num_beam_groups=1, pad_token_id=-3, 
                bos_token_id=-1, eos_token_id=-2, max_new_tokens=12, renormalize_logits=True)
        else:
            generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        normalizer_path: Optional[Union[str, Path]] = None,
        init_rstates: Optional[np.ndarray] = None,
        gym_env_id: Optional[str] = "SmartDryerEnv-v1",
        **kwargs,
    ) -> torch.LongTensor:
        # TODO put original docstring back in after finished with editing
        # 0. Initiate oracle simulation environment and normalizer for RL
        self.env = gym.make(gym_env_id)
        self.env.unwrapped.seed(0)
        # Set policy to inference mode
        self.policy.set_training_mode(False)
        if normalizer_path is None:
            logger.warning("Normalizer path is empty. Please confirm the model doesn't require a normalizer.")
            self.normalizer = None
        else:
            self.normalizer = AutoNormalizer(load_path=normalizer_path)

        if init_rstates is not None: # TODO implement checks for init_rstates shapes
            if (not isinstance(init_rstates, np.ndarray)) or (init_rstates.shape[0] != init_rstates.size):
                raise ValueError("Initial random states must be either None, or an numpy array of shape (ENV_NUM_RSTATES,) for n runs")
            self.env.reset(options={"rstates": init_rstates.squeeze().item() if init_rstates.size == 1 else init_rstates})
        else:
            self.env.reset()

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        # if gym_env_id == "CydrumsEnv-v3":
        #     generation_config.early_stopping = True
        self.generation_config = generation_config # TODO stopgap solution for not being able to find self.generation_config

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = False
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs, init_rstates
        )
        batch_size = input_ids.shape[0]

        # 4. Define other model kwargs
        # 5. Prepare `input_ids` which will be used for auto-regressive generation

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=input_ids,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        # if streamer is not None and (generation_config.num_beams > 1):
        #     raise ValueError(
        #         "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        #     )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            raise NotImplementedError
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            result = self._greedy_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            raise NotImplementedError
        elif generation_mode == GenerationMode.SAMPLE:
            raise NotImplementedError
        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size, # totally expecting this to be always 1
                num_beams=generation_config.num_beams,
                num_actions=ENV_NUM_ACTION_COMBINATIONS,
                device=input_ids.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping, # This makes shorter sequences more likely to be kept
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
                gym_env_id=gym_env_id
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=False, # RL models should be treated as decoder only
                **model_kwargs,
            )
            # 13. run beam search
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                this_rstate=init_rstates,
                synced_gpus=synced_gpus,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            raise NotImplementedError
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            raise NotImplementedError
        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                num_actions=ENV_NUM_ACTION_COMBINATIONS,
                constraints=final_constraints,
                device=input_ids.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
                gym_env_id=gym_env_id
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=False, # RL models should be treated as decoder only
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                this_rstate=init_rstates,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        else:
            raise NotImplementedError

        # if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        #     if not callable(getattr(self, "_reset_cache", None)):
        #         raise ValueError(
        #             "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
        #             " Make sure this model implements a `_reset_cache` function."
        #         )
        #     self._reset_cache()

        return result

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

        # 2. Min length warnings due to unfeasible parameter combinations
        min_length_error_suffix = (
            " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
            "increase the maximum length."
        )
        if has_default_max_length:
            min_length_error_suffix += (
                f" Note that `max_length` is set to {generation_config.max_length}, its default value."
            )
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )
        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length
            if min_length > generation_config.max_length:
                warnings.warn(
                    f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                    f"added to the prompt length ({input_ids_length}), is larger than"
                    f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                    UserWarning,
                )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        # elif (
        #     model_input_name == "inputs_embeds"
        #     and input_ids_length != inputs_tensor.shape[1]
        #     and not self.config.is_encoder_decoder
        # ):
        #     generation_config.max_length -= inputs_tensor.shape[1]

        # same for min length
        if generation_config.min_new_tokens is not None:
            if not has_default_min_length:
                logger.warning(
                    f"Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(="
                    f"{generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        # elif (
        #     model_input_name == "inputs_embeds"
        #     and input_ids_length != inputs_tensor.shape[1]
        #     and not self.config.is_encoder_decoder
        # ):
        #     generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        # if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
        #     processors.append(
        #         UnbatchedClassifierFreeGuidanceLogitsProcessor(
        #             generation_config.guidance_scale,
        #             self,
        #             unconditional_ids=negative_prompt_ids,
        #             unconditional_attention_mask=negative_prompt_attention_mask,
        #             use_cache=model_kwargs["use_cache"],
        #         )
        #     )
        # if generation_config.sequence_bias is not None:
        #     processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        # if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        #     processors.append(
        #         HammingDiversityLogitsProcessor(
        #             diversity_penalty=generation_config.diversity_penalty,
        #             num_beams=generation_config.num_beams,
        #             num_beam_groups=generation_config.num_beam_groups,
        #         )
        #     )
        # if (
        #     generation_config.encoder_repetition_penalty is not None
        #     and generation_config.encoder_repetition_penalty != 1.0
        # ):
        #     processors.append(
        #         EncoderRepetitionPenaltyLogitsProcessor(
        #             penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
        #         )
        #     )
        # if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        #     processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        # if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
        #     processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        # if (
        #     generation_config.encoder_no_repeat_ngram_size is not None
        #     and generation_config.encoder_no_repeat_ngram_size > 0
        # ):
        #     processors.append(
        #         EncoderNoRepeatNGramLogitsProcessor(generation_config.encoder_no_repeat_ngram_size, encoder_input_ids)
        #     )
        # if generation_config.bad_words_ids is not None:
        #     processors.append(
        #         NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
        #     )
        # if (
        #     generation_config.min_length is not None
        #     and generation_config.eos_token_id is not None
        #     and generation_config.min_length > 0
        # ):
        #     processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        # if (
        #     generation_config.min_new_tokens is not None
        #     and generation_config.eos_token_id is not None
        #     and generation_config.min_new_tokens > 0
        # ):
        #     processors.append(
        #         MinNewTokensLengthLogitsProcessor(
        #             input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
        #         )
        #     )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
                )
            )
        # if generation_config.forced_bos_token_id is not None:
        #     processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        # if generation_config.forced_eos_token_id is not None:
        #     processors.append(
        #         ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
        #     )
        # if generation_config.remove_invalid_values is True:
        #     processors.append(InfNanRemoveLogitsProcessor())
        # if generation_config.exponential_decay_length_penalty is not None:
        #     processors.append(
        #         ExponentialDecayLengthPenalty(
        #             generation_config.exponential_decay_length_penalty,
        #             generation_config.eos_token_id,
        #             input_ids_seq_length,
        #         )
        #     )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        # if generation_config.begin_suppress_tokens is not None:
        #     begin_index = input_ids_seq_length
        #     begin_index = (
        #         begin_index
        #         if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
        #         else begin_index + 1
        #     )
        #     if generation_config.forced_decoder_ids is not None:
        #         # generation starts after the last token that is forced
        #         begin_index += generation_config.forced_decoder_ids[-1][0]
        #     processors.append(
        #         SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
        #     )
        # if generation_config.forced_decoder_ids is not None:
        #     # TODO(Sanchit): deprecate in v4.40 by removing this logic
        #     warnings.warn(
        #         "You have explicitly specified `forced_decoder_ids`. This functionality has been deprecated and will throw an error in v4.40. Please remove the `forced_decoder_ids` argument in favour of `input_ids` or `decoder_input_ids` respectively.",
        #         FutureWarning,
        #     )
        #     processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids, _has_warned=True))
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _get_stopping_criteria(
        self, generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            # max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            max_position_embeddings = None
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model._greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        # if "inputs_embeds" in model_kwargs:
        #     cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs # TODO this version is slow, try using env.step() to avoid repetitious calculations
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            unfinished_sequences = unfinished_sequences & ~model_inputs["all_dones"] # unfinished so far && not done at this step
            # forward pass to get next token
            next_token_logits = self.get_next_token_logits(**model_inputs)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # next_tokens_scores = logits_processor(input_ids, next_token_logits) # TODO worry about this later
            next_tokens_scores = next_token_logits

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                raise NotImplementedError
                # if output_scores:
                #     scores += (next_tokens_scores,)
                # if output_logits:
                #     raw_logits += (next_token_logits,)
                # if output_attentions:
                #     decoder_attentions += (
                #         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                #     )
                #     if self.config.is_encoder_decoder:
                #         cross_attentions += (outputs.cross_attentions,)

                # if output_hidden_states:
                #     decoder_hidden_states += (
                #         (outputs.decoder_hidden_states,)
                #         if self.config.is_encoder_decoder
                #         else (outputs.hidden_states,)
                #     )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            # if streamer is not None:
            #     streamer.put(next_tokens.cpu())
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs,
            #     model_kwargs,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            # )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            raise NotImplementedError
            # if self.config.is_encoder_decoder:
            #     return GenerateEncoderDecoderOutput(
            #         sequences=input_ids,
            #         scores=scores,
            #         logits=raw_logits,
            #         encoder_attentions=encoder_attentions,
            #         encoder_hidden_states=encoder_hidden_states,
            #         decoder_attentions=decoder_attentions,
            #         cross_attentions=cross_attentions,
            #         decoder_hidden_states=decoder_hidden_states,
            #         past_key_values=model_kwargs.get("past_key_values"),
            #     )
            # else:
            #     return GenerateDecoderOnlyOutput(
            #         sequences=input_ids,
            #         scores=scores,
            #         logits=raw_logits,
            #         attentions=decoder_attentions,
            #         hidden_states=decoder_hidden_states,
            #         past_key_values=model_kwargs.get("past_key_values"),
            #     )
        else:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            return input_ids, model_inputs

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
        if synced_gpus:
            raise NotImplementedError
            # # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # # The following logic allows an early break if all peers finished generating their sequence
            # this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # # send 0.0 if we finished, 1.0 otherwise
            # dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # # did all peers finish? the reduced sum will be 0.0 then
            # if this_peer_finished_flag.item() == 0.0:
            #     return False
        elif this_peer_finished:
            return False
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # all_obs, all_rewards, all_dones = self.env.cached_batch_rollout(input_ids)
        all_obs, all_rewards, all_dones = self.env.unwrapped.redis_cached_batch_rollout(input_ids)
        if self.normalizer is not None:
            normalized_obs = self.normalizer.normalize_obs(all_obs)
        else:
            normalized_obs = all_obs
        return {
            "input_ids": normalized_obs.to(input_ids.device),
            "all_rewards": all_rewards.to(input_ids.device),
            "all_dones": all_dones.to(input_ids.device),
            "use_cache": False
        }

    @staticmethod
    def _expand_multidiscrete_logits(dist: MultiCategoricalDistribution) -> torch.Tensor:
        """
        Expand the logits of a MultiCategoricalDistribution. Currently, only action_dims == 2 are supported.

        Args:
            dist (MultiCategoricalDistribution): The MultiCategoricalDistribution object.

        Returns:
            torch.Tensor: The expanded logits tensor.
        """
        with torch.no_grad():
            logits_0 = dist.distribution[0].logits
            logits_1 = dist.distribution[1].logits
            # expanded_logits = torch.flatten(logits_0.T.repeat(1, dist.action_dims[1]) + logits_1.repeat(dist.action_dims[0], 1)).unsqueeze(0)
            expanded_logits = torch.flatten(logits_0.unsqueeze(1).repeat(1, dist.action_dims[1], 1).permute(0, 2, 1) + logits_1.unsqueeze(1).repeat(1, dist.action_dims[0], 1), start_dim=1, end_dim=2)
        return expanded_logits

    def get_next_token_logits(self, input_ids: Union[np.ndarray, torch.Tensor], **kwargs):
        """
        Get the policy distribution from an observation.

        :param input_ids: the input observation. If normalization is needed, it
            need to be done before passing to this function.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # self.policy.set_training_mode(False)

        if isinstance(input_ids, torch.Tensor):
            model_input = input_ids.to(self.policy.device)
        else:
            # model_input, vectorized_env = self.policy.obs_to_tensor(input_ids)
            model_input, _ = self.policy.obs_to_tensor(input_ids)

        with torch.no_grad():
            dist = self.policy.get_distribution(model_input)

        # If dist is a MultiCategoricalDistribution, return flattened logits
        if isinstance(dist, MultiCategoricalDistribution):
            expanded_logits = self._expand_multidiscrete_logits(dist)
        else:
            expanded_logits = dist.distribution.logits

        if isinstance(input_ids, torch.Tensor):
            return expanded_logits.to(input_ids.device)
        return expanded_logits # outputs should be of shape (batch_size, vocab_size)

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        this_rstate: Optional[np.ndarray] = None,
        synced_gpus: bool = False,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        sequential = sequential if sequential is not None else self.generation_config.low_memory
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # Not expecting this to get triggered
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        # Reset the oracle environment with the supplied initial conditions
        _, _ = self.env.reset(options={"rstates": this_rstate.squeeze().item() if this_rstate.size == 1 else this_rstate})
        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # find the indices of the beams that have failed
            failed_beam_indices = torch.where(model_inputs['all_rewards'] < 0)[0]
            num_failed_beams = failed_beam_indices.numel()
            if num_failed_beams:
                if num_failed_beams == num_beams:
                    # all beams have failed, no need to continue
                    this_peer_finished = True
                logger.warning("Detected %s failed beams", num_failed_beams)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            # else:  # Unchanged original behavior
            # outputs = self(**model_inputs, return_dict=True, 
            #                output_attentions=output_attentions, 
            #                output_hidden_states=output_hidden_states)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = self.get_next_token_logits(**model_inputs)
            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size) # TODO check performance with or without this line

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                raise NotImplementedError("Dict return is not yet supported for beam_search")
                # if output_scores:
                #     scores += (next_token_scores_processed,)
                # if output_logits:
                #     raw_logits += (next_token_logits,)
                # if output_attentions:
                #     decoder_attentions += (
                #         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                #     )
                #     if self.config.is_encoder_decoder:
                #         cross_attentions += (outputs.cross_attentions,)
                # if output_hidden_states:
                #     decoder_hidden_states += (
                #         (outputs.decoder_hidden_states,)
                #         if self.config.is_encoder_decoder
                #         else (outputs.hidden_states,)
                #     )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            # next_token_scores, next_tokens = torch.topk(
            #     next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            # )
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, min(2816, vocab_size*num_beams), dim=1, largest=True, sorted=True
            ) # Capping the max. number of tokens to 2816 (64 beams * 44 vocabulary size) to prevent unlimited growth of candidates

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                oracle_env_rstates=this_rstate,
                rewards_array=model_inputs["all_rewards"],
                this_done_array=model_inputs["all_dones"],
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs,
            #     model_kwargs,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            # )
            # if model_kwargs.get("past_key_values", None) is not None:
            #     model_kwargs["past_key_values"] = self._temporary_reorder_cache(
            #         model_kwargs["past_key_values"], beam_idx
            #     )

            # if return_dict_in_generate and output_scores:
            #     beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1
            # torch.sum(model_inputs['all_dones']).item() > 0 indicates at least one beam has finished
            # if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)) or torch.sum(model_inputs['all_dones']).item() > 0:
            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
            max_refine_steps=BS_MAX_REFINE_STEPS,
            logits_processor=logits_processor,
        )

        if return_dict_in_generate:
            raise NotImplementedError("Dict return is not yet supported for beam_search")
        else:
            model_inputs = self.prepare_inputs_for_generation(sequence_outputs["sequences"], **model_kwargs)
            # Only print the best result as my result parsing script only supports one result
            best_idx = torch.argmax(model_inputs['all_rewards']).item()
            selected_model_inputs = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    selected_model_inputs[k] = v[best_idx].unsqueeze(0)
                else:
                    selected_model_inputs[k] = v
            return sequence_outputs["sequences"][best_idx], selected_model_inputs

    def _constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        this_rstate: Optional[np.ndarray] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._constrained_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     ConstrainedBeamSearchScorer,
        ...     PhrasalConstraint,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> constraint_str = "Sie"
        >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
        >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


        >>> # instantiate beam scorer
        >>> beam_scorer = ConstrainedBeamSearchScorer(
        ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model._constrained_beam_search(
        ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt sind Sie?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You have not defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else False
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # Not expecting this to get triggered
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        # Reset the oracle environment with the supplied initial conditions
        _, _ = self.env.reset(options={"rstates": this_rstate.squeeze().item() if this_rstate.size == 1 else this_rstate})
        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # outputs = self(
            #     **model_inputs,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )

            # find the indices of the beams that have failed
            failed_beam_indices = torch.where(model_inputs['all_rewards'] < 0)[0]
            num_failed_beams = failed_beam_indices.numel()
            if num_failed_beams:
                if num_failed_beams == num_beams:
                    # all beams have failed, no need to continue
                    this_peer_finished = True
                logger.warning("Detected %s failed beams", num_failed_beams)
            # if num_failed_beams:
            #     logger.warning(f"Detected {num_failed_beams} failed beams. Attempting to replace them.")
            #     # find beams that have not failed
            #     replace_beam_indices = torch.where(model_inputs['all_rewards'] >= -100)[0]
            #     num_remaining_beams = replace_beam_indices.numel()
            #     if num_remaining_beams == 0:
            #         # if all beams have failed, break the loop
            #         logger.warning("All beams have failed. Aborting generation.")
            #         break
            #     # create a copy of input_ids
            #     input_ids_back = input_ids.clone()
            #     beam_scores_back = beam_scores.clone()
            #     for i, fb in enumerate(failed_beam_indices):
            #         # set these beams to one of the beams from replace_beam_indices
            #         input_ids[fb, :] = input_ids_back[replace_beam_indices[i % num_remaining_beams], :]
            #         # copy the beam scores to the failed beams location
            #         beam_scores[fb] = beam_scores_back[replace_beam_indices[i % num_remaining_beams]] - 1
            #     # rerun self.prepare_inputs_for_generation to update the model_inputs
            #     model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = self.get_next_token_logits(**model_inputs)
            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # A heuristic to prune the input_ids that is known to have failed due to simenv numerical instability
            # (e.g. model_inputs['all_rewards'] < -100 returned by self.prepare_inputs_for_generation(input_ids, **model_kwargs))
            # This can be done by setting the next_token_scores to -inf for those beams in RLCBSDiscardFailedSequencesLogitsProcessor
            # As logits processor call signatures are fixed, we need to manipulate the inputs to signal the failed beams
            # If model_inputs['all_rewards'] < -100, set the first token of the corresponding beam to -66
            # # find the indices of the beams that have failed
            # failed_beam_indices = torch.where(model_inputs['all_rewards'] < -100)[0]
            # # set the first token of the failed beams to -66
            # input_ids[failed_beam_indices, 0] = -66

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                raise NotImplementedError("Dict return is not yet supported for constrained_beam_search")

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            # next_token_scores, next_tokens = torch.topk(
            #     next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            # ) # Original logic for determining k could impact performance by limiting the number of tokens passed to the beam scorer
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, min(2816, vocab_size*num_beams), dim=1, largest=True, sorted=True
            ) # Capping the max. number of tokens to 2816 (64 beams * 44 vocabulary size) to prevent unlimited growth of candidates

            next_indices = (next_tokens / vocab_size).long()
            # next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor") # TODO compare old/new version for next_indices
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                oracle_env_rstates=this_rstate,
                rewards_array=model_inputs["all_rewards"],
                this_done_array=model_inputs["all_dones"],
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs,
            #     model_kwargs,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            # )
            # if model_kwargs.get("past_key_values", None) is not None:
            #     model_kwargs["past_key_values"] = self._temporary_reorder_cache(
            #         model_kwargs["past_key_values"], beam_idx
            #     )

            # if return_dict_in_generate and output_scores:
            #     beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if constrained_beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
            max_refine_steps=BS_MAX_REFINE_STEPS,
            logits_processor=logits_processor,
        )

        if return_dict_in_generate:
            raise NotImplementedError("Dict return is not yet supported for constrained_beam_search")
        else:
            model_inputs = self.prepare_inputs_for_generation(sequence_outputs["sequences"], **model_kwargs)
            # Only print the best result as my result parsing script only supports one result
            best_idx = torch.argmax(model_inputs['all_rewards']).item()
            selected_model_inputs = {}
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    selected_model_inputs[k] = v[best_idx].unsqueeze(0)
                else:
                    selected_model_inputs[k] = v
            return sequence_outputs["sequences"][best_idx], selected_model_inputs
