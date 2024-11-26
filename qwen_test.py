# env for the gate agent to decide which expert to choose
# following the gym env interface
import sys
from typing import Any, Dict, Optional, Union
sys.path.append('/home/aibench/ting_storage/github_clone/Expert_Sparsity')
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache, StaticCache, Cache
# from model.wrapper_qwen import DynamicSkippingQwenSparseMoeBlockWrapper
from transformers.modeling_outputs import(
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    WatermarkLogitsProcessor,
)

from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

# from transformers.utils import (
#     is_torchdynamo_compiling
# )
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.utils import GenerationMixin
# from transformers.logits_process import(
#     LogitsProcessorList
# )
class Inference():

    def __init__(self, previous_net=None, qoe_weights=None, log_path=None, num_experts=60, net_window=10, hidden_states=None,
                num_activated_expert=None, initial_expert_weights=None, compute_power=None, num_device=10, tokenizer=None,
                model=None, llm=None, num_layer=24, top_k=4, render_mode = None) -> None:
        self.num_device = num_device
        self.net_window = net_window
        self.previous_network = previous_net
        self.qoe_weights = qoe_weights
        self.log_path = log_path
        self.num_experts = num_experts
        self.num_activated_expert = num_activated_expert
        self.initial_expert_weights = initial_expert_weights
        self.compute_power = compute_power
        self.hidden_states = hidden_states
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.model = model
        self.llm = llm
        self.before_gate_modules = []
        self.after_gate_modules = []
        self.num_layer = num_layer
        self.cur_layer = 0
        self.verbose = True
        self.kwargs = None
        self.residual = None
        self.last_hidden_states = None
        self.past_key_value = DynamicCache()
        self.router_logits = None
        self.present_key_value = None
        self.self_attn_weights = None
        self.do_sample = True
        self.token_over = False
        self.max_length = 100
        self.cur_len = 0
        self.input_ids = None
        self.all_self_attns = ()
        self.all_router_logits = ()
        self.next_decoder_cache  = None
        self.generation_config = GenerationConfig()
        self.answer = []
        self.model_inputs = None

        self.state = None

        # eos related generation config 
        self.generation_config.max_time = None
        self.generation_config.stop_strings = None
        # kwarg
        self.stopping_criteria = None
        self._eos_token_tensor = None
        self.max_position_embeddings = None
        self.generation_config.bos_token_id = 151643
        self.generation_config.pad_token_id = 151643
        self.generation_config.eos_token_id = [151643, 151645]

        # PPO related parameters
        self.render_mode = None 
        self._max_episode_steps = 1000
        self.query = None

    def assign_module(self):
        # assign the before_gate_module and after_gate_module
        for i in range(len(self.model.model.layers)):
            self.after_gate_module = {}
            self.after_gate_module['experts'] = self.model.model.layers[i].mlp.experts
            self.after_gate_module['shared_expert'] = self.model.model.layers[i].mlp.shared_expert
            self.after_gate_module['shared_expert_gate'] = self.model.model.layers[i].mlp.shared_expert_gate
            self.after_gate_module['input_layernorm'] = self.model.model.layers[i].input_layernorm
            self.after_gate_module['post_attention_layernorm'] = self.model.model.layers[i].post_attention_layernorm
    
            self.after_gate_modules.append(self.after_gate_module)
            self.before_gate_modules.append(self.model.model.layers[self.cur_layer].self_attn)
    
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )


    def _has_unfinished_sequences(
            self,
            this_peer_finished: bool,
            synced_gpus: bool,
            device: torch.device,
            cur_len: Optional[int] = None,
            max_length: Optional[int] = None,
        ) -> bool:
            """
            Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
            fed through `this_peer_finished`. ZeRO stage 3-friendly.
            """
            # torch.compile does not support data-dependent control flow. This is a workaround to allow torch.compile,
            # although we lose the ability to stop when all sequences return an EOS token (and other stopping criteria)
            if is_torchdynamo_compiling(): 
                return cur_len < max_length
            else:
                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        return False
                elif this_peer_finished:
                    return False
                return True

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"

        return cache_name, past_key_values
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
        device: str,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config._eos_token_tensor, list):
                min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
            elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
                min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            warpers.append(MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(
                    epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                )
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
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



    def init(self, seed):
        # dynamic assign query for trainning
        # chosen_query = np.random.choice(self.query)
        chosen_query = self.query[0]

        inputs = self.tokenizer(chosen_query, return_tensors="pt")
        input_ids = inputs["input_ids"]
        self.input_ids = input_ids
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        # hidden_states = self.before_gate_modules[0](inputs_embeds)
        # self.hidden_states = hidden_states
        self.current_token_length = hidden_states.shape[1]

        # generate kwargs
        self.kwargs = {
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
            "position_ids": torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids),
            "past_key_value": DynamicCache(),
            "output_attentions": False,
            "use_cache": True,
            "cache_position": torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device),
            "output_router_logits": False,
        }
        self.kwargs = self._get_initial_cache_position(input_ids, self.kwargs)
        self.residual = hidden_states

        hidden_states = self.model.model.layers[0].input_layernorm(hidden_states)

        # Ensure attention_mask is 2D
        if self.kwargs["attention_mask"].dim() > 2:
            self.kwargs["attention_mask"] = self.kwargs["attention_mask"].squeeze(1).squeeze(1)

        self.kwargs['position_ids'] = None
        self.model_inputs = self.prepare_inputs_for_generation(self.input_ids, **self.kwargs)

        # Self Attention
        hidden_states, self_attn_weights, past_key_value = self.model.model.layers[self.cur_layer].self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=self.kwargs['position_ids'],
            past_key_value=self.kwargs['past_key_value'],
            output_attentions=self.kwargs['output_attentions'],
            use_cache=self.kwargs['use_cache'],
            cache_position=self.kwargs['cache_position'],
            # output_router_logits=self.kwargs["output_router_logits"],
        )
        hidden_states = self.residual + hidden_states

        # Fully Connected
        self.residual = hidden_states
        self.past_key_value = past_key_value
        self.self_attn_weights = self_attn_weights
        hidden_states = self.model.model.layers[0].post_attention_layernorm(hidden_states)

        self.hidden_states = hidden_states 
        return self.state


    def step(self): 
        over = False # self.model.model.layers[0].mlp.gate
        #! begin moe inference
        # expert_weight = np.nonzero(action)[0]
        # note that the hidden state here is the tensor that has been passed through the self attention layer
        batch_size, sequence_length, hidden_dim = self.hidden_states.shape
        hidden_states = self.hidden_states.view(-1, hidden_dim)

        router_logits= self.model.model.layers[self.cur_layer].mlp.gate(hidden_states)

        # router_logits = torch.from_numpy(np.ones((batch_size * sequence_length, self.num_experts)))
        # obtain the original router logits
        # router_logits = router_logits_origin # debug

        # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # postive n 60
        routing_weights = F.softmax(router_logits, dim=0, dtype=torch.float) # Debug for env

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # (n, 4) & (n, 4)
        # if self.norm_topk_prob:
        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitat
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        # expert_mask = expert_mask.permute(1, 0)


        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.model.model.layers[self.cur_layer].mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx]) # 1 4 n 

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            middel_state = routing_weights[top_x, idx, None]
            current_hidden_states = expert_layer(current_state) * middel_state

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.after_gate_modules[self.cur_layer]['shared_expert'](hidden_states)
        shared_expert_output = F.sigmoid(self.after_gate_modules[self.cur_layer]['shared_expert_gate'](hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        moe_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        #! end moe inference

        #! begin between moe and self attention
        if isinstance(moe_hidden_states, tuple):
            moe_hidden_states, router_logits = moe_hidden_states
        else:
            router_logits = None
        
        hidden_states = self.residual + moe_hidden_states

        layer_outputs = (hidden_states,)

        if self.kwargs['output_attentions']:
            layer_outputs += (self_attn_weights,)

        if self.kwargs['use_cache']:
            layer_outputs += (self.present_key_value,)

        if self.kwargs['output_router_logits']:
            layer_outputs += (router_logits,)
        #! end between moe and self attention

        # start the judging of whether the eos is reached
        if self.cur_layer + 1 < self.num_layer:
            self.token_over = False
            self.hidden_states = hidden_states
            self.cur_layer += 1
            # print('pass a layer')
        else:
            # finish one token generation
            self.cur_layer = 0
            self.token_over = True
            hidden_states = self.model.model.norm(hidden_states)
            self.last_hidden_states = hidden_states
            # decode the output of the model
            logits = self.model.lm_head(hidden_states)
            logits = logits.float()
            outputs = MoeCausalLMOutputWithPast(
                logits=logits,
                past_key_values=self.next_decoder_cache,
                hidden_states=None,
                attentions=self.all_self_attns if self.kwargs['output_attentions'] else None,
                router_logits=self.all_router_logits if self.kwargs['output_router_logits'] else None,
            )

            # init eos related tokens
            def _tensor_or_none(token, device=None):
                if token is None:
                    return token

                device = device if device is not None else self.device
                if isinstance(token, torch.Tensor):
                    return token.to(device)
                return torch.tensor(token, device=device, dtype=torch.long)

            # init tensors with tokens
            self._pad_token_tensor = _tensor_or_none(self.generation_config.pad_token_id, device='cuda')
            self._eos_token_tensor = _tensor_or_none(self.generation_config.eos_token_id, device='cuda')
            # print("eos_token_tensor: ",self._eos_token_tensor)
            self._bos_token_tensor = _tensor_or_none(self.generation_config.bos_token_id, device='cuda')

            if self._eos_token_tensor is not None and self._eos_token_tensor.ndim == 0:
                self._eos_token_tensor = self._eos_token_tensor.unsqueeze(0)

            # Set pad token if unset (and there are conditions to do so)
            if self._pad_token_tensor is None and self._eos_token_tensor is not None:
                kwargs_has_attention_mask = self.kwargs["attention_mask"] is not None
                
                if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                    # logger.warning(
                    #     "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    #     "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                    # )
                    pass
                self._pad_token_tensor = self._eos_token_tensor[0]
                # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")
            pad_token_id = self._pad_token_tensor # debug    

            # 9. prepare stopping criteria
            stopping_criteria = self.stopping_criteria if self.stopping_criteria is not None else StoppingCriteriaList()
            criteria = StoppingCriteriaList()
            
            if self.max_length is not None:
                # max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                criteria.append(
                    MaxLengthCriteria(
                        max_length=self.max_length,
                        max_position_embeddings=self.max_position_embeddings,
                    )
                )
            if self.generation_config.max_time is not None:
                criteria.append(MaxTimeCriteria(max_time=self.generation_config.max_time))
            if self.generation_config.stop_strings is not None:
                if tokenizer is None:
                    raise ValueError(
                        "There are one or more stop strings, either in the arguments to `generate` or in the "
                        "model's generation config, but we could not locate a tokenizer. When generating with "
                        "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                    )
                criteria.append(StopStringCriteria(stop_strings=self.generation_config.stop_strings, tokenizer=tokenizer))
            if self._eos_token_tensor is not None:
                criteria.append(EosTokenCriteria(eos_token_id=self._eos_token_tensor))

            stopping_criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
            # print("stopping_criteria: ",stopping_criteria)
            # stopping_criteria = criteria

            # finished sentences should have their next token be a padding token
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria) # debug
            # print("has_eos_stopping_criteria: ",has_eos_stopping_criteria)       

            # init unfinished_sequences
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device='cuda') # debug
            this_peer_finished = False

            # 进入transformers中generationmixin的判断是否结束的循环-->self._has_unfinished_sequences
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            # next_token_scores = next_token_logits
            logits_processor = LogitsProcessorList()
            next_token_scores = logits_processor(self.input_ids, next_token_logits)
            # logits_warper = None
            
            # qwen_moe generation config
            self.generation_config.do_sample = True
            # self.generation_config.bos_token_id = 151643
            # self.generation_config.pad_token_id = 151643
            # self.generation_config.eos_token_id = [151645, 151643]
            prepared_logits_warper = (
                self._get_logits_warper(self.generation_config, device=self.input_ids.device)
                if self.generation_config.do_sample
                else None
            )
            if self.do_sample:
                logits_warper = prepared_logits_warper
                next_token_scores = logits_warper(self.input_ids, next_token_scores)
            
            # token selection
            if self.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            # print('the next token is', self.tokenizer.decode(next_tokens))
            # print('the next token id is', next_tokens)
            self.answer.append(self.tokenizer.decode(next_tokens))
            # finished sentences should have their next token be a padding token
            # has_eos_stopping_criteria = False # debug
            # pad_token_id = 151643 # debug
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            self.input_ids = self.input_ids.to(next_tokens.device)
            
            ## hidden state length varified
            self.input_ids = torch.cat([self.input_ids, next_tokens[:, None]], dim=-1)
            streamer = None # debug
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            self.kwargs = self._update_model_kwargs_for_generation(
                outputs,
                self.kwargs,
                is_encoder_decoder=False,
            )
            
            scores = 0 # debug
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(self.input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            self.cur_len += 1
            
            del outputs

            # judge whether is the eos meet
            if is_torchdynamo_compiling():
                over = (self.cur_len < self.max_length)
            else:
                # check whether the trace is finished
                over = False
                synced_gpus = False
                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    device = 'cuda'
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        over =  True
                elif this_peer_finished:
                    over =  True
                
                if over == False:
                    # prepare model inputs
                    self.kwargs['position_ids'] = None
                    self.model_inputs = self.prepare_inputs_for_generation(self.input_ids, **self.kwargs)

                    # prepare variable output controls (note: some models won't accept all output controls)
                    # set to false the same as default qwen moe inference
                    output_attentions = False
                    output_hidden_states = False
                    self.model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                    self.model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if self.token_over == True:
            use_legacy_cache = False
            if self.model_inputs['use_cache'] and not isinstance(self.model_inputs['past_key_values'], Cache):
                use_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(self.model_inputs['past_key_values'])
                # logger.warning_once(
                #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
                # )

            input_embeds = self.model.model.embed_tokens(self.model_inputs['input_ids'])
        
            # causal_mask = self._update_causal_mask(
            #     model_inputs['attention_mask'], inputs_embeds, model_inputs['cache_position'],
            #     model_inputs['past_key_values'], output_attentions)
            causal_mask = None
            hidden_states = input_embeds
            
            # decoder layers
            # all_hidden_states = () if output_hidden_states else None
            # all_self_attns = () if output_attentions else None
            # all_router_logits = None
            self.next_decoder_cache = None



        if not over:
            #! begin before self attention
            if self.token_over == True:
                hidden_states = hidden_states
            else:
                hidden_states = layer_outputs[0]
                if self.kwargs['use_cache']:
                    next_decoder_cache = layer_outputs[2 if self.kwargs['output_attentions'] else 1]
                    self.next_decoder_cache = next_decoder_cache

                if self.kwargs['output_attentions']:
                    self.all_self_attns += (layer_outputs[1],)

                if self.kwargs['output_router_logits'] and layer_outputs[-1] is not None:
                    self.all_router_logits += (layer_outputs[-1],)   

            residual = hidden_states
            hidden_states = self.model.model.layers[self.cur_layer].input_layernorm(hidden_states)
            
            hidden_states, self_attn_weights, present_key_value = self.model.model.layers[self.cur_layer].self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=self.model_inputs['position_ids'],
                past_key_value=self.model_inputs['past_key_values'],
                output_attentions=self.kwargs['output_attentions'],
                use_cache=self.model_inputs['use_cache'],
                cache_position=self.model_inputs['cache_position'],
            )

            hidden_states = residual + hidden_states 
            residual = hidden_states
            hidden_states = self.model.model.layers[self.cur_layer].post_attention_layernorm(hidden_states)

            # update self information
            self.hidden_states = hidden_states
            self.residual = residual
            self.self_attn_weights = self_attn_weights
            self.present_key_value = present_key_value

        return over



# begin the test of the env 
if __name__ == '__main__':
    queries = [
    "This is a dairy. This is a good day. I want to play with my friends.",
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "How does a neural network work?",
    "What are the benefits of using renewable energy?",
    "Can you summarize the plot of 'Pride and Prejudice'?",
    "What is the difference between supervised and unsupervised learning?",
    "How do quantum computers differ from classical computers?",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "What are the key features of Python programming language?"
    ]

    model_path = '/home/aibench/ting_storage/aimodels/qwenmoe'
    use_flash_attention_2 = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # load_in_8bit_fp32_cpu_offload=True  
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cuda',
        # device_map="balanced",  
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        # quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else None
    )

    inference_instant = Inference(model=model, tokenizer=tokenizer)
    inference_instant.assign_module()
    inference_instant.query = queries

    seed = 777
    import time
    # for i in range(50):
    seed += 1
    state = inference_instant.init(seed)

    step_number = 1024

    for j in range(step_number):
        action = torch.rand(1, 19, 60).to('cuda')
        over= inference_instant.step()
        # print('step', j)
        if over:
            print('finish the token generation')
            print('the answer of the model is', inference_instant.answer)
            if inference_instant.verbose == True:
                pass
            break

    print('the answer of the model is', inference_instant.answer)