"""Declared execution boundaries preserve row identity and globally weighted gradients."""

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches


def batch(sizes=None):
    value = TensorDict({'row': torch.arange(5).unsqueeze(1)}, batch_size=[5])
    tu.assign_non_tensor(value, use_dynamic_bsz=False, micro_batch_size_per_gpu=2)
    if sizes is not None:
        tu.assign_non_tensor(value, micro_batch_sizes=sizes)
    return value


@pytest.mark.parametrize('sizes', [(2, 2, 1), (1, 3, 1), (5,)])
def test_declared_partial_tail_preserves_global_gradient_and_one_update(sizes):
    packs, permutation = prepare_micro_batches(batch(sizes))
    assert permutation is None
    assert [len(pack) for pack in packs] == list(sizes)
    assert torch.cat([pack['row'] for pack in packs]).flatten().tolist() == list(range(5))
    parameter = torch.tensor(0.7, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    inputs = torch.tensor([1., 2., 3., 4., 5.])
    weights = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])
    for pack in packs:
        indices = pack['row'].flatten()
        (weights[indices] * (parameter * inputs[indices]).square()).sum().backward()
    expected_gradient = 2 * 0.7 * (weights * inputs.square()).sum()
    torch.testing.assert_close(parameter.grad, expected_gradient)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor(0.7) - 0.1 * expected_gradient)


@pytest.mark.parametrize('sizes', [(), (2, 2), (2, 2, 2), (0, 5), (-1, 6), (True, 4), (2.5, 2.5)])
def test_declared_invalid_coverage_rejected(sizes):
    with pytest.raises(ValueError):
        prepare_micro_batches(batch(sizes))


def test_legacy_uniform_divisibility_is_unchanged():
    with pytest.raises(AssertionError, match='divisible'):
        prepare_micro_batches(batch())


def test_declared_boundaries_reject_rebalancing_and_group_splits():
    value = batch((2, 2, 1))
    tu.assign_non_tensor(value, use_dynamic_bsz=True)
    with pytest.raises(ValueError, match='dynamic'):
        prepare_micro_batches(value)
    tu.assign_non_tensor(value, use_dynamic_bsz=False, force_group_size=2)
    with pytest.raises(ValueError, match='force_group_size'):
        prepare_micro_batches(value)


def test_declared_boundaries_retain_count_constraints():
    with pytest.raises(ValueError, match='divisibility'):
        prepare_micro_batches(batch((2, 2, 1)), num_batches_divided_by=2)
    with pytest.raises(ValueError, match='minimum'):
        prepare_micro_batches(batch((2, 2, 1)), min_num_micro_batch=4)


def _distributed_partition_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        # Different local pack sizes are valid when collective counts agree.
        packs, _ = prepare_micro_batches(batch((2, 2, 1) if rank == 0 else (1, 1, 3)))
        assert len(packs) == 3
        # Every rank sees the invalid declaration before starting backward.
        with pytest.raises(ValueError, match='counts differ'):
            prepare_micro_batches(batch((2, 2, 1) if rank == 0 else (5,)))
        with pytest.raises(ValueError, match='cover every row'):
            prepare_micro_batches(batch((2, 2, 1) if rank == 0 else (1, 1, 1)))
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_declared_boundaries_validate_all_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_partition_worker,
                               args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
