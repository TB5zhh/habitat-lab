import enum


class EnvironmentWorkerTasks(enum.Enum):
    start = enum.auto()
    step = enum.auto()
    reset = enum.auto()
    set_transfer_buffers = enum.auto()
    set_action_plugin = enum.auto()
    start_experience_collection = enum.auto()
    wait = enum.auto()


class ReportWorkerTasks(enum.Enum):
    episode_end = enum.auto()
    learner_update = enum.auto()
    learner_timing = enum.auto()
    env_timing = enum.auto()
    policy_timing = enum.auto()
    start_collection = enum.auto()
    state_dict = enum.auto()
    load_state_dict = enum.auto()
    preemption_decider = enum.auto()
    num_steps_collected = enum.auto()


class PreemptionDeciderTasks(enum.Enum):
    policy_step = enum.auto()
    start_rollout = enum.auto()
    end_rollout = enum.auto()
    learner_time = enum.auto()
