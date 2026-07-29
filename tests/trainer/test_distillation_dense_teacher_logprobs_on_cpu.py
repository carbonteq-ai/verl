import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import _teacher_log_probs_to_response


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
