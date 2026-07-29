from omegaconf import OmegaConf

from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import Role


def _config(*, enable_resource_pool: bool):
    return OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "reward": {
                "reward_model": {
                    "enable": False,
                    "enable_resource_pool": False,
                    "n_gpus_per_node": 0,
                    "nnodes": 0,
                }
            },
            "distillation": {
                "enabled": True,
                "enable_resource_pool": enable_resource_pool,
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
        }
    )


def test_colocated_teacher_reuses_global_resource_pool():
    config = _config(enable_resource_pool=False)
    runner = BaseTaskRunner()

    runner.add_teacher_model_resource_pool(config)
    manager = runner.init_resource_pool_mgr(config)

    assert runner.mapping[Role.TeacherModel] == "global_pool"
    assert manager.resource_pool_spec == {"global_pool": [1]}
    assert config.distillation.n_gpus_per_node == config.trainer.n_gpus_per_node
    assert config.distillation.nnodes == config.trainer.nnodes


def test_dedicated_teacher_keeps_separate_resource_pool():
    config = _config(enable_resource_pool=True)
    runner = BaseTaskRunner()

    runner.add_teacher_model_resource_pool(config)
    manager = runner.init_resource_pool_mgr(config)

    assert runner.mapping[Role.TeacherModel] == "teacher_pool"
    assert manager.resource_pool_spec == {
        "global_pool": [1],
        "teacher_pool": [1],
    }
