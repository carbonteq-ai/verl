import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    _teacher_log_probs_to_response,
    compute_distillation_loss_range,
)


def test_dense_teacher_log_probs_are_sliced_to_padded_responses() -> None:
    data = TensorDict(
        {
            "prompts": torch.tensor(
                [
                    [101, 102, 103, 104],
                    [0, 0, 201, 202],
                ]
            ),
            "responses": torch.tensor(
                [
                    [105, 106, 107],
                    [203, 204, 0],
                ]
            ),
            # vLLM prompt log probabilities omit the first input token and
            # append one dummy row, so each response begins at prompt_width - 1.
            "teacher_logprobs": torch.tensor(
                [
                    [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0], [0.0]],
                    [[0.0], [0.0], [-7.0], [-8.0], [-9.0], [0.0], [0.0]],
                ]
            ),
        },
        batch_size=[2],
    )

    actual = _teacher_log_probs_to_response(data)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [-4.0, -5.0, -6.0],
                [-8.0, -9.0, 0.0],
            ]
        ),
    )


def test_nested_teacher_log_probs_use_logical_micro_batch_rows() -> None:
    full_batch = torch.nested.as_nested_tensor(
        [
            torch.tensor([[-10.0], [-11.0], [-12.0]]),
            torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]]),
            torch.tensor([[-6.0], [-7.0], [-8.0], [-9.0]]),
            torch.tensor([[-13.0], [-14.0], [-15.0]]),
        ],
        layout=torch.jagged,
    )
    teacher_micro_batch = torch.nested.nested_tensor_from_jagged(
        full_batch.values(),
        offsets=torch.tensor([3, 8, 12]),
        lengths=torch.tensor([5, 4]),
    )
    assert teacher_micro_batch.values().shape[0] > sum(
        sample.shape[0] for sample in teacher_micro_batch.unbind()
    )
    data = TensorDict(
        {
            "prompts": torch.tensor(
                [
                    [101, 102, 103],
                    [0, 201, 202],
                ]
            ),
            "responses": torch.tensor(
                [
                    [104, 105],
                    [203, 204],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                ]
            ),
            "teacher_logprobs": teacher_micro_batch,
        },
        batch_size=[2],
    )

    actual = _teacher_log_probs_to_response(data)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [-3.0, -4.0],
                [-7.0, -8.0],
            ]
        ),
    )


def test_distillation_loss_range_ignores_fully_masked_padding_micro_batch() -> None:
    assert (
        compute_distillation_loss_range(
            distillation_losses=torch.tensor([[1.0]]),
            response_mask=torch.tensor([[False]]),
        )
        == {}
    )
