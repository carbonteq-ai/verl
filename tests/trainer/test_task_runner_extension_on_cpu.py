"""Recipe extension does not need Ray's private actor metadata."""

from verl.trainer.main_ppo import TaskRunnerV1, TaskRunnerV1Base


def test_plain_task_runner_base_can_be_specialized_without_starting_ray():
    class RecipeRunner(TaskRunnerV1Base):
        def run(self, config):
            self.config = config
            return "recipe"

    runner = RecipeRunner()
    assert runner.config is None and runner.trainer is None and runner.agent_loop_manager is None
    assert runner.run("config") == "recipe"
    assert runner.config == "config"
    assert RecipeRunner.init_agent_loop_manager is TaskRunnerV1Base.init_agent_loop_manager
    assert callable(TaskRunnerV1.remote)
