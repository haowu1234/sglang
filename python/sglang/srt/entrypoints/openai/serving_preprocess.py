from __future__ import annotations

import dataclasses
import enum
import pickle
from array import array
from typing import Any, Dict, Iterable, List, Optional

import msgspec

from sglang.srt.entrypoints.openai.protocol import (
    OpenAIServingRequest,
    PreprocessAdapterInfo,
    PreprocessMultimodalInfo,
    PreprocessRequestItem,
    PreprocessResponse,
    PreprocessReturnOptions,
    PreprocessRoutingInfo,
    PreprocessSessionInfo,
)
from sglang.srt.managers.io_struct import PickleWrapper, TokenizedGenerateReqInput

_MM_TOKEN_ID_FIELDS = (
    "im_token_id",
    "im_start_id",
    "im_end_id",
    "slice_start_id",
    "slice_end_id",
    "video_token_id",
    "audio_token_id",
    "audio_start_id",
    "audio_end_id",
)

_MM_SUMMARY_FIELDS = (
    "mrope_positions",
    "mrope_position_delta",
    "vision_position_ids",
    "media_nums_per_sample",
    "visible_frame_counts",
)


def build_preprocess_response(
    tokenized_requests: Iterable[TokenizedGenerateReqInput],
    request: OpenAIServingRequest,
    source_endpoint: str,
    adapted_request: Any = None,
) -> PreprocessResponse:
    return_prompt_by_rid = _return_prompt_token_ids_by_rid(adapted_request)
    return PreprocessResponse(
        source_endpoint=source_endpoint,
        model=getattr(request, "model", None),
        requests=[
            _build_preprocess_item(
                tokenized_request,
                return_prompt_token_ids=return_prompt_by_rid.get(
                    tokenized_request.rid, False
                ),
            )
            for tokenized_request in tokenized_requests
        ],
    )


def _build_preprocess_item(
    tokenized_request: TokenizedGenerateReqInput,
    return_prompt_token_ids: bool,
) -> PreprocessRequestItem:
    input_ids = _array_to_list(tokenized_request.input_ids)
    return PreprocessRequestItem(
        rid=tokenized_request.rid,
        input_text=_json_safe(tokenized_request.input_text),
        input_ids=input_ids,
        input_token_count=len(input_ids) if input_ids is not None else 0,
        token_type_ids=_json_safe(tokenized_request.token_type_ids),
        sampling_params=_sampling_params_to_dict(tokenized_request.sampling_params),
        return_options=PreprocessReturnOptions(
            return_logprob=tokenized_request.return_logprob,
            logprob_start_len=tokenized_request.logprob_start_len,
            top_logprobs_num=tokenized_request.top_logprobs_num,
            return_prompt_token_ids=return_prompt_token_ids,
            return_hidden_states=tokenized_request.return_hidden_states,
            require_reasoning=tokenized_request.require_reasoning,
        ),
        routing=PreprocessRoutingInfo(
            routed_dp_rank=tokenized_request.routed_dp_rank,
            disagg_prefill_dp_rank=tokenized_request.disagg_prefill_dp_rank,
            routing_key=tokenized_request.routing_key,
            priority=tokenized_request.priority,
            extra_key=tokenized_request.extra_key,
            bootstrap_host=tokenized_request.bootstrap_host,
            bootstrap_port=tokenized_request.bootstrap_port,
            bootstrap_room=tokenized_request.bootstrap_room,
        ),
        adapter=PreprocessAdapterInfo(lora_id=tokenized_request.lora_id),
        session=PreprocessSessionInfo(
            session_id=tokenized_request.session_id,
            session_params=_json_safe(tokenized_request.session_params),
        ),
        multimodal=_multimodal_info(tokenized_request.mm_inputs),
    )


def _sampling_params_to_dict(sampling_params: Any) -> Dict[str, Any]:
    if sampling_params is None:
        return {}
    if isinstance(sampling_params, msgspec.Struct):
        return _json_safe(msgspec.structs.asdict(sampling_params))
    return _json_safe(sampling_params)


def _return_prompt_token_ids_by_rid(adapted_request: Any) -> Dict[str, bool]:
    if adapted_request is None:
        return {}
    if getattr(adapted_request, "is_single", True):
        return {
            adapted_request.rid: bool(
                getattr(adapted_request, "return_prompt_token_ids", False)
            )
        }
    if isinstance(getattr(adapted_request, "rid", None), list):
        return {
            adapted_request.rid[i]: bool(
                getattr(adapted_request[i], "return_prompt_token_ids", False)
            )
            for i in range(len(adapted_request.rid))
        }
    return {}


def _multimodal_info(mm_inputs: Any) -> PreprocessMultimodalInfo:
    mm_inputs = _maybe_unwrap_pickle(mm_inputs)
    if mm_inputs is None:
        return PreprocessMultimodalInfo()

    token_ids = {
        field: _json_safe(getattr(mm_inputs, field))
        for field in _MM_TOKEN_ID_FIELDS
        if getattr(mm_inputs, field, None) is not None
    }
    summaries = {
        field: _summarize_tensor_like(getattr(mm_inputs, field))
        for field in _MM_SUMMARY_FIELDS
        if getattr(mm_inputs, field, None) is not None
    }

    return PreprocessMultimodalInfo(
        has_mm_inputs=True,
        items=[
            _multimodal_item_to_dict(item)
            for item in getattr(mm_inputs, "mm_items", []) or []
        ],
        padded_input_ids=_json_safe(getattr(mm_inputs, "padded_input_ids", None)),
        token_ids=token_ids,
        summaries=summaries,
    )


def _multimodal_item_to_dict(item: Any) -> Dict[str, Any]:
    ret = {
        "modality": _enum_name(getattr(item, "modality", None)),
        "hash": _json_safe(getattr(item, "hash", None)),
        "pad_value": _json_safe(getattr(item, "pad_value", None)),
        "offsets": _json_safe(getattr(item, "offsets", None)),
        "format": _enum_name(getattr(item, "format", None)),
    }
    model_specific_data = getattr(item, "model_specific_data", None)
    if model_specific_data:
        ret["model_specific_keys"] = sorted(model_specific_data.keys())

    return {k: v for k, v in ret.items() if v is not None}


def _array_to_list(value: Optional[array]) -> Optional[List[int]]:
    if value is None:
        return None
    return list(value)


def _maybe_unwrap_pickle(value: Any) -> Any:
    if isinstance(value, PickleWrapper):
        return pickle.loads(value.data)
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, array):
        return list(value)
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, msgspec.Struct):
        return _json_safe(msgspec.structs.asdict(value))
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _summarize_tensor_like(value)
    return str(value)


def _summarize_tensor_like(value: Any) -> Dict[str, Any]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        shape = [int(dim) for dim in shape]
    return {
        "shape": shape,
        "dtype": str(getattr(value, "dtype", None)),
    }


def _enum_name(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.name
    return _json_safe(value)
