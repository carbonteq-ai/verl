from verl.workers.rollout.llm_server import (
    compute_spec_decode_counter_delta,
    parse_spec_decode_prometheus,
)


def test_parse_spec_decode_prometheus_sums_labeled_counter_series():
    snapshot = parse_spec_decode_prometheus(
        """
# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.
vllm:spec_decode_num_drafts_total{engine="0"} 10
vllm:spec_decode_num_draft_tokens_total{engine="0"} 20
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 15
vllm:spec_decode_num_drafts_total{engine="1"} 2
vllm:spec_decode_num_draft_tokens_total{engine="1"} 4
vllm:spec_decode_num_accepted_tokens_total{engine="1"} 1
"""
    )

    assert snapshot == {"drafts": 12.0, "draft_tokens": 24.0, "accepted_tokens": 16.0}


def test_spec_decode_counter_delta_is_step_local_and_handles_engine_reset():
    first, snapshot = compute_spec_decode_counter_delta(
        {"drafts": 10.0, "draft_tokens": 20.0, "accepted_tokens": 15.0}, {}
    )
    second, snapshot = compute_spec_decode_counter_delta(
        {"drafts": 16.0, "draft_tokens": 32.0, "accepted_tokens": 24.0}, snapshot
    )
    after_reset, _ = compute_spec_decode_counter_delta(
        {"drafts": 2.0, "draft_tokens": 4.0, "accepted_tokens": 1.0}, snapshot
    )

    assert first["rollout/spec_num_draft_tokens"] == 20.0
    assert first["rollout/spec_accept_rate"] == 0.75
    assert first["rollout/spec_accept_length"] == 2.5
    assert second["rollout/spec_num_draft_tokens"] == 12.0
    assert second["rollout/spec_num_accepted_tokens"] == 9.0
    assert second["rollout/spec_accept_rate"] == 0.75
    assert after_reset["rollout/spec_num_draft_tokens"] == 4.0
    assert after_reset["rollout/spec_num_accepted_tokens"] == 1.0
