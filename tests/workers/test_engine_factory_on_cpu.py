"""Recipe engine construction retains native configuration and dispatch setup."""

from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
from verl.workers import engine_workers  # noqa: E402


@pytest.mark.parametrize("specialized", [False, True])
def test_training_worker_engine_factory_before_dispatch(monkeypatch, specialized):
    calls, dispatch = [], []
    engine = SimpleNamespace(get_data_parallel_rank=lambda: 0, is_mp_src_rank_with_outputs=lambda: True)
    config = SimpleNamespace(model_type="language_model",
        model_config=SimpleNamespace(get=lambda key, default: default),
        engine_config=SimpleNamespace(strategy="fsdp"), optimizer_config=object(),
        checkpoint_config=object(), profiler_config=None)
    monkeypatch.setattr(engine_workers.Worker, "__init__", lambda self: None)
    monkeypatch.setattr(engine_workers.Worker, "rank", property(lambda self: 0))
    monkeypatch.setattr(engine_workers, "initialize_global_process_group_ray", lambda **kwargs: None)
    monkeypatch.setattr(engine_workers, "set_numa_affinity", lambda: None)
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(engine_workers, "DistProfiler", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(engine_workers.DistProfilerExtension, "__init__", lambda self, profiler: None)
    monkeypatch.setattr(engine_workers.TrainingWorker, "_register_dispatch_collect_info",
        lambda self, **kwargs: dispatch.append((self.engine, kwargs)))

    from verl.workers.engine import EngineRegistry

    def create(**kwargs):
        calls.append(kwargs)
        return engine

    monkeypatch.setattr(EngineRegistry, "new", create)
    if specialized:
        class RecipeWorker(engine_workers.TrainingWorker):
            def create_engine(self):
                calls.append({"recipe": self.config.model_type})
                return engine

        worker_type = RecipeWorker
    else:
        worker_type = engine_workers.TrainingWorker
    worker = worker_type(config)
    assert worker.engine is engine
    assert len(calls) == 1
    if specialized:
        assert calls[0] == {"recipe": "language_model"}
    else:
        assert calls[0] == dict(model_type="language_model", backend="fsdp",
            model_config=config.model_config, engine_config=config.engine_config,
            optimizer_config=config.optimizer_config, checkpoint_config=config.checkpoint_config)
    assert dispatch == [(engine, dict(mesh_name="train", dp_rank=0, is_collect=True))]
    assert worker.loss_fn is None
