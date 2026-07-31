from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio  # noqa: E402
import unittest  # noqa: E402
from array import array  # noqa: E402
from unittest.mock import AsyncMock, Mock  # noqa: E402

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.utils import get_or_create_event_loop  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _AsyncReaderLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ModelUpdateLock:
    def __init__(self):
        self.reader_lock = _AsyncReaderLock()


def _tokenized(rid, input_ids=None):
    params = SamplingParams(max_new_tokens=1)
    params.normalize(None)
    params.verify(32000)
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="prompt",
        input_ids=array("q", input_ids or [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=params,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def _manager():
    manager = object.__new__(TokenizerManager)
    manager.auto_create_handle_loop = Mock()
    manager.enable_priority_scheduling = False
    manager.default_priority_value = None
    manager.elastic_worker_count = 1
    manager.enable_trace = False
    manager.disaggregation_mode = DisaggregationMode.NULL
    manager.rid_to_state = {}
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = _ModelUpdateLock()
    manager.server_args = Mock(
        enable_tokenizer_batch_encode=False,
        enable_dp_attention=False,
    )
    manager._tokenize_one_request = AsyncMock(
        side_effect=lambda obj: _tokenized(obj.rid)
    )
    manager._send_one_request = Mock()
    manager._send_batch_request = Mock()
    manager._wait_one_response = Mock()
    manager._handle_epd_disaggregation_encode_request = Mock()
    manager.enable_lora = False
    manager.lora_registry = Mock()
    return manager


class TestTokenizerManagerPreprocess(unittest.TestCase):
    def setUp(self):
        self.loop = get_or_create_event_loop()

    def test_preprocess_tokenizes_without_scheduler_dispatch(self):
        manager = _manager()
        req = GenerateReqInput(text="hello", sampling_params={"max_new_tokens": 1})

        tokenized = self.loop.run_until_complete(
            manager.preprocess_generate_request(req)
        )

        self.assertEqual(len(tokenized), 1)
        self.assertEqual(tokenized[0].input_ids.tolist(), [1, 2, 3])
        manager._tokenize_one_request.assert_awaited_once()
        manager._send_one_request.assert_not_called()
        manager._send_batch_request.assert_not_called()
        manager._wait_one_response.assert_not_called()
        manager._handle_epd_disaggregation_encode_request.assert_not_called()
        self.assertEqual(manager.rid_to_state, {})

    def test_preprocess_cleans_req_state_on_tokenization_error(self):
        manager = _manager()
        manager._tokenize_one_request = AsyncMock(side_effect=ValueError("bad input"))
        req = GenerateReqInput(text="hello", sampling_params={"max_new_tokens": 1})

        with self.assertRaisesRegex(ValueError, "bad input"):
            self.loop.run_until_complete(manager.preprocess_generate_request(req))

        self.assertEqual(manager.rid_to_state, {})
        manager._send_one_request.assert_not_called()

    def test_preprocess_releases_lora_after_success_and_error(self):
        for should_raise in [False, True]:
            manager = _manager()
            manager.enable_lora = True
            manager.lora_registry.release = AsyncMock()

            async def resolve_lora(obj):
                obj.lora_id = "lora-1"

            manager._validate_and_resolve_lora = AsyncMock(side_effect=resolve_lora)
            if should_raise:
                manager._tokenize_one_request = AsyncMock(
                    side_effect=ValueError("tokenize failed")
                )

            req = GenerateReqInput(
                text="hello",
                sampling_params={"max_new_tokens": 1},
                lora_path="adapter",
            )
            if should_raise:
                with self.assertRaisesRegex(ValueError, "tokenize failed"):
                    self.loop.run_until_complete(
                        manager.preprocess_generate_request(req)
                    )
            else:
                self.loop.run_until_complete(manager.preprocess_generate_request(req))

            manager.lora_registry.release.assert_awaited_once_with("lora-1")
            self.assertEqual(manager.rid_to_state, {})

    def test_preprocess_returns_expanded_parallel_samples(self):
        manager = _manager()
        req = GenerateReqInput(
            text="hello", sampling_params={"max_new_tokens": 1, "n": 3}
        )

        tokenized = self.loop.run_until_complete(
            manager.preprocess_generate_request(req)
        )

        rids = [item.rid for item in tokenized]
        self.assertEqual(len(rids), 3)
        self.assertEqual(len(set(rids)), 3)
        self.assertEqual(manager._tokenize_one_request.await_count, 3)
        self.assertEqual(manager.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
