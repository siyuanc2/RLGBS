# coding=utf-8
# This file is derived from HuggingFace Transformers project
# (https://github.com/huggingface/transformers).
# Licensed under the Apache License, Version 2.0.
# 
# Copyright 2022 The HuggingFace Inc. team.
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
""" Generation configuration class and utilities."""

import copy
import json
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from .hf_logging import get_logger

logger = get_logger(__name__)
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")

__version__ = "4.40.0.dev0"

class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"

class GenerationConfig():
    # no-format
    r"""
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and
            `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0.`
            and `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and
            `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and
            `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if
            `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if
            `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if
            `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if
            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to '.generate()'. To learn
    more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.

    </Tip>

    Arg:
        > Parameters that control the length of the output

        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to 0):
            The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
            `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
        min_new_tokens (`int`, *optional*):
            The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        max_time(`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still finish
            the current pass after allocated time has been passed.

        > Parameters that control the generation strategy used

        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        penalty_alpha (`float`, *optional*):
            The values balance the model confidence and the degeneration penalty in contrastive search decoding.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        typical_p (`float`, *optional*, defaults to 1.0):
            Local typicality measures how similar the conditional probability of predicting a target token next is to
            the expected conditional probability of predicting a random token next, given the partial text already
            generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
            add up to `typical_p` or higher are kept for generation. See [this
            paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        epsilon_cutoff (`float`, *optional*, defaults to 0.0):
            If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
            `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
            size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        eta_cutoff (`float`, *optional*, defaults to 0.0):
            Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
            0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
            exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
            probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
            depending on the size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        encoder_repetition_penalty (`float`, *optional*, defaults to 1.0):
            The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
            original input. 1.0 means no penalty.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids(`List[List[int]]`, *optional*):
            List of list of token ids that are not allowed to be generated. Check
            [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
        force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of
            words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this
            triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
            can allow different forms of each word.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or warpers (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors or warpers break the normalization.
        constraints (`List[Constraint]`, *optional*):
            Custom constraints that can be added to the generation to ensure that the output will contain the use of
            certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`Union[int, List[int]]`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
        remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
            penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens  (`List[int]`, *optional*):
            A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their
            log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`List[int]`, *optional*):
            A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit
            processor will set their log probs to `-inf` so that they are not sampled.
        forced_decoder_ids (`List[List[int]]`, *optional*):
            A list of pairs of integers which indicates a mapping from generation indices to token indices that will be
            forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token
            of index 123.
        sequence_bias (`Dict[Tuple[int], float]`, *optional*)):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. Check
            [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
        guidance_scale (`float`, *optional*):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        low_memory (`bool`, *optional*):
            Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory.
            Used with beam search and contrastive search.


        > Parameters that define the output variables of `generate`

        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*):
            Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        > Generation parameters exclusive to encoder-decoder models

        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        decoder_start_token_id (`Union[int, List[int]]`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
            `batch_size`. Indicating a list enables different start ids for each element in the batch
            (e.g. multilingual models with different target languages in one batch)


        > Generation parameters exclusive to [assistant generation](https://arxiv.org/abs/2211.17192)

        num_assistant_tokens (`int`, *optional*, defaults to 5):
            Defines the number of _speculative tokens_ that shall be generated by the assistant model before being
            checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
            more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant
            model requires lots of corrections, lower speed-ups are reached.

        num_assistant_tokens_schedule (`str`, *optional*, defaults to `"heuristic"`):
            Defines the schedule at which max assistant tokens shall be changed during inference.
            - `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
              reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
            - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
            - `"constant"`: `num_assistant_tokens` stays unchanged during generation

        prompt_lookup_num_tokens (`int`, *optional*, default to `None`):
            The number of tokens to be output as candidate tokens.

        max_matching_ngram_size (`int`, *optional*, default to `None`):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.

        > Parameters specific to the caching mechanism:

        cache_implementation (`str`, *optional*, default to `None`):
            Cache class that should be used when generating.


        > Wild card

        generation_kwargs:
            Additional generation kwargs will be forwarded to the `generate` function of the model. Kwargs that are not
            present in `generate`'s signature will be used in the model forward pass.
    """

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.use_cache = kwargs.pop("use_cache", True)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 5)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "heuristic")

        # Cache implementation
        self.cache_implementation = kwargs.pop("cache_implementation", None)

        # Prompt lookup decoding
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters not validated here are best validated at generate runtime, as they may depend on
        other inputs and/or the model, such as parameters related to the generation length.

        Arg:
            is_init (`bool`, *optional*, defaults to `False`):
                Whether the validation is performed during the initialization of the instance.
        """

        # Validation of individual attributes
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}.")

        # Validation of attribute relations:
        fix_location = ""
        if is_init:
            fix_location = (
                " This was detected when initializing the generation config instance, which means the corresponding "
                "file may hold incorrect parameterization and should be fixed."
            )

        # 1. detect sampling-only parameterization when not in sampling mode
        if self.do_sample is False:
            greedy_wrong_parameter_msg = (
                "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only "
                "used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
                + fix_location
            )
            if self.temperature is not None and self.temperature != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="temperature", flag_value=self.temperature),
                    UserWarning,
                )
            if self.top_p is not None and self.top_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p),
                    UserWarning,
                )
            if self.typical_p is not None and self.typical_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="typical_p", flag_value=self.typical_p),
                    UserWarning,
                )
            if (
                self.top_k is not None and self.top_k != 50 and self.penalty_alpha is None
            ):  # contrastive search uses top_k
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k),
                    UserWarning,
                )
            if self.epsilon_cutoff is not None and self.epsilon_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff),
                    UserWarning,
                )
            if self.eta_cutoff is not None and self.eta_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="eta_cutoff", flag_value=self.eta_cutoff),
                    UserWarning,
                )

        # 2. detect beam-only parameterization when not in beam mode
        if self.num_beams is None:
            warnings.warn("`num_beams` is set to None - defaulting to 1.", UserWarning)
            self.num_beams = 1

        if self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to 1. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used "
                "in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`." + fix_location
            )
            if self.early_stopping is not False:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="early_stopping", flag_value=self.early_stopping),
                    UserWarning,
                )
            if self.num_beam_groups is not None and self.num_beam_groups != 1:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="num_beam_groups", flag_value=self.num_beam_groups
                    ),
                    UserWarning,
                )
            if self.diversity_penalty is not None and self.diversity_penalty != 0.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="diversity_penalty", flag_value=self.diversity_penalty
                    ),
                    UserWarning,
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="length_penalty", flag_value=self.length_penalty),
                    UserWarning,
                )
            if self.constraints is not None:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="constraints", flag_value=self.constraints),
                    UserWarning,
                )

        # 3. detect incorrect paramaterization specific to advanced beam modes
        else:
            # constrained beam search
            if self.constraints is not None or self.force_words_ids is not None:
                constrained_wrong_parameter_msg = (
                    "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. However, "
                    "`{flag_name}` is set to `{flag_value}`, which is incompatible with this generation mode. Set "
                    "`constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue." + fix_location
                )
                if self.do_sample is True:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(flag_name="do_sample", flag_value=self.do_sample)
                    )
                if self.num_beam_groups is not None and self.num_beam_groups != 1:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(
                            flag_name="num_beam_groups", flag_value=self.num_beam_groups
                        )
                    )
            # group beam search
            if self.diversity_penalty != 0.0 or self.num_beam_groups != 1:
                group_error_prefix = (
                    "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In "
                    "this generation mode, "
                )
                if self.do_sample is True:
                    raise ValueError(group_error_prefix + "`do_sample` must be set to `False`")
                if self.num_beams % self.num_beam_groups != 0:
                    raise ValueError(group_error_prefix + "`num_beams` should be divisible by `num_beam_groups`")
                if self.diversity_penalty == 0.0:
                    raise ValueError(
                        group_error_prefix
                        + "`diversity_penalty` should be greater than `0.0`, otherwise your groups will be identical."
                    )

        # 4. check `num_return_sequences`
        if self.num_return_sequences != 1:
            if self.num_beams == 1:
                if self.do_sample is False:
                    raise ValueError(
                        "Greedy methods without beam search do not support `num_return_sequences` different than 1 "
                        f"(got {self.num_return_sequences})."
                    )
            elif self.num_return_sequences > self.num_beams:
                raise ValueError(
                    f"`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` "
                    f"({self.num_beams})."
                )

        # 5. check common issue: passing `generate` arguments inside the generation config
        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
                )

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # The line below allows model-specific config to be loaded as well through kwargs, with safety checks.
        # See https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Fields to ignore at serialization time
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        # Transformers version when serializing this file
        output["transformers_version"] = __version__

        self.dict_torch_dtype_to_str(output)
        return output

    def to_json_string(self, use_diff: bool = True, ignore_metadata: bool = False) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing atributtes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Confirm that the updated instance is still valid
        self.validate()

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
