from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import json  # noqa: E402
import pickle  # noqa: E402
import unittest  # noqa: E402
from array import array  # noqa: E402
from unittest.mock import AsyncMock, Mock, patch  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import Request  # noqa: E402

from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    CompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.srt.entrypoints.openai.serving_completions import (  # noqa: E402
    OpenAIServingCompletion,
)
from sglang.srt.entrypoints.openai.serving_preprocess import (  # noqa: E402
    build_preprocess_response,
)
from sglang.srt.managers.io_struct import (  # noqa: E402
    PickleWrapper,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.utils import get_or_create_event_loop  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name = "llama-3"
        self.jinja_template_content_format = None
        self.completion_template_name = None
        self.reasoning_config = None
        self.force_reasoning = False


def _sampling_params(**kwargs):
    params = SamplingParams(**kwargs)
    tokenizer = Mock()
    tokenizer.encode.return_value = [42]
    params.normalize(tokenizer)
    params.verify(32000)
    return params


def _tokenized(rid="rid-1", input_ids=None, **kwargs):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=kwargs.pop("input_text", "prompt"),
        input_ids=array("q", input_ids or [1, 2, 3]),
        input_embeds=None,
        mm_inputs=kwargs.pop("mm_inputs", None),
        token_type_ids=kwargs.pop("token_type_ids", None),
        sampling_params=kwargs.pop("sampling_params", _sampling_params()),
        return_logprob=kwargs.pop("return_logprob", False),
        logprob_start_len=kwargs.pop("logprob_start_len", -1),
        top_logprobs_num=kwargs.pop("top_logprobs_num", 0),
        token_ids_logprob=kwargs.pop("token_ids_logprob", None),
        stream=kwargs.pop("stream", False),
        return_hidden_states=kwargs.pop("return_hidden_states", False),
        require_reasoning=kwargs.pop("require_reasoning", False),
        routed_dp_rank=kwargs.pop("routed_dp_rank", None),
        disagg_prefill_dp_rank=kwargs.pop("disagg_prefill_dp_rank", None),
        routing_key=kwargs.pop("routing_key", None),
        priority=kwargs.pop("priority", None),
        extra_key=kwargs.pop("extra_key", None),
        bootstrap_host=kwargs.pop("bootstrap_host", None),
        bootstrap_port=kwargs.pop("bootstrap_port", None),
        bootstrap_room=kwargs.pop("bootstrap_room", None),
        lora_id=kwargs.pop("lora_id", None),
        session_id=kwargs.pop("session_id", None),
        session_params=kwargs.pop("session_params", None),
    )


def _tokenizer_manager(tokenized):
    manager = Mock()
    manager.model_config = Mock(is_multimodal=False)
    manager.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
    manager.model_config.context_len = 32000
    manager.server_args = Mock(
        enable_cache_report=False,
        tool_call_parser="hermes",
        reasoning_parser=None,
        stream_response_default_include_usage=False,
        default_chat_template_kwargs=None,
        context_length=32000,
        allow_auto_truncate=False,
    )
    manager.tokenizer = Mock()
    manager.tokenizer.encode.return_value = [1, 2, 3]
    manager.tokenizer.decode.return_value = "decoded"
    manager.tokenizer.chat_template = None
    manager.tokenizer.bos_token_id = 1
    manager.request_logger = Mock(log_requests=False, log_requests_level=0)
    manager.preprocess_generate_request = AsyncMock(return_value=[tokenized])
    return manager


class TestOpenAIServingPreprocess(unittest.TestCase):
    def setUp(self):
        self.raw_request = Mock(spec=Request)
        self.raw_request.headers = {}
        self.loop = get_or_create_event_loop()

    def test_completion_preprocess_uses_internal_conversion(self):
        tokenized = _tokenized(
            input_ids=[11, 22],
            sampling_params=_sampling_params(max_new_tokens=7, temperature=0.0),
            routed_dp_rank=3,
            routing_key="route-a",
        )
        manager = _tokenizer_manager(tokenized)
        serving = OpenAIServingCompletion(manager, _MockTemplateManager())
        self.raw_request.headers = {
            "x-data-parallel-rank": "3",
            "x-smg-routing-key": "route-a",
        }

        req = CompletionRequest(
            model="x",
            prompt="hello",
            max_tokens=7,
            temperature=0.0,
            routed_dp_rank=1,
        )
        response = self.loop.run_until_complete(
            serving.handle_preprocess_request(
                req, self.raw_request, source_endpoint="/v1/completions"
            )
        )

        body = json.loads(response.body)
        self.assertEqual(body["object"], "sglang.preprocess")
        self.assertEqual(body["version"], "v1")
        self.assertEqual(body["source_endpoint"], "/v1/completions")
        self.assertEqual(body["requests"][0]["input_ids"], [11, 22])
        self.assertEqual(body["requests"][0]["sampling_params"]["max_new_tokens"], 7)

        adapted = manager.preprocess_generate_request.await_args.args[0]
        self.assertEqual(adapted.routed_dp_rank, 3)
        self.assertEqual(adapted.routing_key, "route-a")

    def test_chat_preprocess_reuses_message_processing(self):
        tokenized = _tokenized(input_ids=[101, 102])
        manager = _tokenizer_manager(tokenized)
        serving = OpenAIServingChat(manager, _MockTemplateManager())
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=5,
        )

        with patch.object(serving, "_process_messages") as process_messages:
            process_messages.return_value = MessageProcessingResult(
                prompt="rendered",
                prompt_ids=[101, 102],
                image_data=None,
                video_data=None,
                audio_data=None,
                modalities=[],
                stop=None,
                tool_call_constraint=None,
            )
            response = self.loop.run_until_complete(
                serving.handle_preprocess_request(
                    req, self.raw_request, source_endpoint="/v1/chat/completions"
                )
            )

        body = json.loads(response.body)
        self.assertEqual(body["source_endpoint"], "/v1/chat/completions")
        self.assertEqual(body["requests"][0]["input_ids"], [101, 102])
        process_messages.assert_called_once()
        manager.preprocess_generate_request.assert_awaited_once()

    def test_streaming_preprocess_is_rejected_before_tokenization(self):
        manager = _tokenizer_manager(_tokenized())
        serving = OpenAIServingCompletion(manager, _MockTemplateManager())
        req = CompletionRequest(model="x", prompt="hello", stream=True)

        response = self.loop.run_until_complete(
            serving.handle_preprocess_request(
                req, self.raw_request, source_endpoint="/v1/completions"
            )
        )

        self.assertEqual(response.status_code, 400)
        manager.preprocess_generate_request.assert_not_called()


class TestPreprocessSerializer(unittest.TestCase):
    def test_sampling_params_and_arrays_are_json_safe(self):
        request = CompletionRequest(model="x", prompt="hello", return_token_ids=True)
        tokenized = _tokenized(
            rid="rid-a",
            input_ids=[5, 6],
            sampling_params=_sampling_params(max_new_tokens=4, stop="END"),
            return_logprob=True,
            top_logprobs_num=2,
        )
        response = build_preprocess_response(
            [tokenized], request, "/v1/completions"
        ).model_dump(mode="json")

        item = response["requests"][0]
        self.assertEqual(item["input_ids"], [5, 6])
        self.assertEqual(item["input_token_count"], 2)
        self.assertEqual(item["sampling_params"]["stop_strs"], ["END"])
        self.assertTrue(item["sampling_params"]["is_normalized"])

    def test_multimodal_metadata_omits_payloads_for_raw_and_pickled_inputs(self):
        mm_output = MultimodalProcessorOutput(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    hash=123,
                    pad_value=456,
                    offsets=[(1, 3)],
                    format=MultimodalInputFormat.NORMAL,
                    feature=np.zeros((2, 2), dtype=np.float32),
                )
            ],
            padded_input_ids=[7, 456, 456, 456, 8],
            im_token_id=999,
            mrope_positions=np.zeros((3, 4), dtype=np.int64),
        )

        for mm_inputs in [mm_output, PickleWrapper(pickle.dumps(mm_output))]:
            request = CompletionRequest(model="x", prompt="hello")
            response = build_preprocess_response(
                [_tokenized(mm_inputs=mm_inputs)], request, "/v1/completions"
            ).model_dump(mode="json")
            mm = response["requests"][0]["multimodal"]
            self.assertTrue(mm["has_mm_inputs"])
            self.assertEqual(mm["items"][0]["modality"], "IMAGE")
            self.assertEqual(mm["items"][0]["hash"], 123)
            self.assertEqual(mm["token_ids"]["im_token_id"], 999)
            self.assertEqual(mm["summaries"]["mrope_positions"]["shape"], [3, 4])
            self.assertNotIn("feature", mm["items"][0])
            self.assertNotIn("precomputed_embeddings", mm["items"][0])


if __name__ == "__main__":
    unittest.main()
