import contextlib
import functools
import os
import time
from collections import defaultdict
from multiprocessing.context import BaseContext
from typing import Any, Dict, Iterator, List, Optional, Tuple

import attr
import numpy as np
import torch

from habitat import Config, logger
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    gather_objects,
    get_distrib_size,
    init_distrib_slurm,
    rank0_only,
)
from habitat_baselines.rl.ver.queue import BatchedQueue
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.worker_common import (
    ProcessBase,
    WindowedRunningMean,
    WorkerBase,
)


@attr.s(auto_attribs=True)
class ReportWorkerProcess(ProcessBase):
    port: int
    config: Config
    report_queue: BatchedQueue
    my_t_zero: float
    num_steps_done: torch.Tensor
    time_taken: torch.Tensor
    n_update_reports: int = 0
    flush_secs: int = 30
    _world_size: Optional[int] = None
    _prev_time_taken: float = 0.0
    timing_stats: Dict = attr.ib(init=False, default=None)
    stats_this_rollout: Dict[str, List[float]] = attr.ib(
        init=False, factory=lambda: defaultdict(list)
    )
    steps_delta: int = 0
    writer: Optional[Any] = None

    def __attrs_post_init__(self):
        self.build_dispatch_table(ReportWorkerTasks)

    def state_dict(self):
        self.response_queue.put(
            dict(
                prev_time_taken=float(self.time_taken),
                window_episode_stats=self.window_episode_stats,
                num_steps_done=int(self.num_steps_done),
                timing_stats=self.timing_stats,
                running_frames_window=self.running_frames_window,
                running_time_window=self.running_time_window,
                n_update_reports=self.n_update_reports,
            )
        )

    def load_state_dict(self, state_dict):
        self._prev_time_taken = state_dict["prev_time_taken"]
        self.time_taken.fill_(self._prev_time_taken)
        self.window_episode_stats = state_dict["window_episode_stats"]
        self.num_steps_done.fill_(int(state_dict["num_steps_done"]))
        self.timing_stats = state_dict["timing_stats"]
        self.running_frames_window = state_dict["running_frames_window"]
        self.running_time_window = state_dict["running_time_window"]
        self.n_update_reports = state_dict["n_update_reports"]

    @property
    def world_size(self) -> int:
        if self._world_size is None:
            if not torch.distributed.is_initialized():
                self._world_size = 1
            else:
                self._world_size = torch.distributed.get_world_size()

        return self._world_size

    def _reduce(self, arr, mean: bool = False):
        if self.world_size == 1:
            return arr

        arr_lens = torch.tensor(
            [[len(arr)]], dtype=torch.int64, device=self.device
        ).repeat(self.world_size, 1)
        torch.distributed.all_gather(list(arr_lens.unbind(0)), arr_lens[0])
        if not torch.all(arr_lens == len(arr)):
            if rank0_only():
                logger.warning(
                    "Skipping reduce as not all arrays are the same size.  "
                    "(If only seen a few times, this indicates that not all workers have had an episode "
                    "finish yet and all is fine.)"
                )

            return arr

        t = torch.from_numpy(arr).to(device=self.device, copy=True)
        torch.distributed.all_reduce(t)
        if mean:
            t.div_(self.world_size)

        return t.cpu().numpy()

    def _reduce_dict(self, d, mean: bool = False):
        if self.world_size == 1 or len(d) == 0:
            return d

        keys = sorted(d.keys())
        stats = self._reduce(
            np.array([d[k] for k in keys], dtype=np.float64), mean=mean
        )

        return {k: type(d[k])(stats[i]) for i, k in enumerate(keys)}

    def _all_reduce(self, val, reduce_op=torch.distributed.ReduceOp.SUM):
        if self.world_size == 1:
            return val

        t = torch.as_tensor(val, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=reduce_op)
        return type(val)(t)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Iterator[Tuple[str, float]]:
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                for subk, subv in cls._extract_scalars_from_info(v):
                    yield f"{k}.{subk}", subv
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                yield k, float(v)

    def get_time(self):
        return time.perf_counter() - self.my_t_zero

    def log_metrics(self, writer: TensorboardWriter, learner_metrics):
        to_reduce = dict()
        to_reduce["count_steps_delta"] = self.steps_delta
        to_reduce["total_time"] = self.get_time() - self.start_time

        to_reduce = self._reduce_dict(to_reduce)

        steps_delta = int(to_reduce["count_steps_delta"])
        self.num_steps_done += steps_delta
        last_time_taken = float(self.time_taken)
        self.time_taken.fill_(
            to_reduce["total_time"] / self.world_size + self._prev_time_taken
        )

        self.running_frames_window += steps_delta
        self.running_time_window += float(self.time_taken) - last_time_taken

        self.steps_delta = 0

        all_stats_this_rollout = gather_objects(
            dict(self.stats_this_rollout), device=self.device
        )
        self.stats_this_rollout.clear()

        learner_metrics = self._reduce_dict(learner_metrics, mean=True)

        preemption_decider_report = self._reduce_dict(
            self.preemption_decider_report
        )

        if rank0_only():
            for stats in all_stats_this_rollout:
                for k, vs in stats.items():
                    self.window_episode_stats[k].add_many(vs)

        if rank0_only():
            n_steps = int(self.num_steps_done)
            if "reward" in self.window_episode_stats:
                writer.add_scalar(
                    "reward",
                    self.window_episode_stats["reward"].mean,
                    n_steps,
                )

            for k in self.window_episode_stats.keys():
                if k == "reward":
                    continue

                writer.add_scalar(
                    f"metrics/{k}", self.window_episode_stats[k].mean, n_steps
                )

            for k, v in learner_metrics.items():
                writer.add_scalar(f"learner/{k}", v, n_steps)

            writer.add_scalar(
                "perf/fps",
                n_steps / float(self.time_taken),
                n_steps,
            )
            writer.add_scalar(
                "perf/fps_window",
                float(self.running_frames_window)
                / float(self.running_time_window),
                n_steps,
            )

        if rank0_only():
            preemption_decider_report = {
                k: v / (self.world_size if "time" in k else 1)
                for k, v in preemption_decider_report.items()
            }
            for k, v in preemption_decider_report.items():
                writer.add_scalar(f"preemption_decider/{k}", v, n_steps)

        if self.n_update_reports % self.config.LOG_INTERVAL == 0:
            if rank0_only():
                logger.info(
                    "update: {}\tfps: {:.1f}\twindow fps: {:.1f}\tframes: {:d}".format(
                        self.n_update_reports,
                        n_steps / float(self.time_taken),
                        float(self.running_frames_window)
                        / float(self.running_time_window),
                        n_steps,
                    )
                )
                if len(self.window_episode_stats) > 0:
                    logger.info(
                        "Average window size: {}  {}".format(
                            list(self.window_episode_stats.values())[0].count,
                            "  ".join(
                                "{}: {:.3f}".format(k, v.mean)
                                for k, v in self.window_episode_stats.items()
                            ),
                        )
                    )
            for name in sorted(self.timing_stats.keys()):
                stats = self._reduce_dict(
                    {k: float(v) for k, v in self.timing_stats[name].items()}
                )
                if rank0_only():
                    logger.info(
                        "{}: ".format(name)
                        + "  ".join(
                            "{}: {:.1f}ms".format(k, v / self.world_size * 1e3)
                            for k, v in sorted(
                                stats.items(),
                                key=lambda kv: kv[1],
                                reverse=True,
                            )
                        )
                    )

    def episode_end(self, data):
        self.stats_this_rollout["reward"].append(data["reward"])
        for k, v in self._extract_scalars_from_info(data["info"]):
            self.stats_this_rollout[k].append(v)

    def num_steps_collected(self, num_steps: int):
        self.steps_delta = num_steps

    def learner_update(self, data):
        self.n_update_reports += 1
        self.log_metrics(self.writer, data)

    def start_collection(self, start_time):
        start_time = self._all_reduce(
            start_time - self.my_t_zero,
            reduce_op=torch.distributed.ReduceOp.MIN,
        )
        self.start_time = start_time

    def preemption_decider(self, preemption_decider_report):
        self.preemption_decider_report = preemption_decider_report

    def env_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["env"][k] += v

    def policy_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["policy"][k] += v

    def learner_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["learner"][k] += v

    @property
    def task_queue(self) -> BatchedQueue:
        return self.report_queue

    def run(self):
        self.device = torch.device("cpu")
        if get_distrib_size()[2] > 1:
            os.environ["MAIN_PORT"] = str(self.port)
            init_distrib_slurm(backend="gloo")
            torch.distributed.barrier()

        self.response_queue.put(None)

        ppo_cfg = self.config.RL.PPO

        self.steps_delta = 0
        if rank0_only():
            self.window_episode_stats = defaultdict(
                functools.partial(
                    WindowedRunningMean,
                    ppo_cfg.reward_window_size
                    * self.config.NUM_ENVIRONMENTS
                    * self.world_size,
                )
            )
        else:
            self.window_episode_stats = None

        timing_types = {
            ReportWorkerTasks.env_timing: "env",
            ReportWorkerTasks.policy_timing: "policy",
            ReportWorkerTasks.learner_timing: "learner",
        }
        self.timing_stats = {
            n: defaultdict(
                functools.partial(
                    WindowedRunningMean, ppo_cfg.reward_window_size
                )
            )
            for n in timing_types.values()
        }
        self.preemption_decider_report = {}

        self.start_time = self.get_time()
        self.running_time_window = WindowedRunningMean(
            ppo_cfg.reward_window_size
        )
        self.running_frames_window = WindowedRunningMean(
            ppo_cfg.reward_window_size
        )

        with (
            get_writer(
                self.config,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            self.writer = writer
            super().run()

            self.writer = None


class ReportWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        port: int,
        config: Config,
        report_queue: BatchedQueue,
        my_t_zero: float,
        init_num_steps=0,
    ):
        self.num_steps_done = torch.full(
            (), int(init_num_steps), dtype=torch.int64
        )
        self.time_taken = torch.full((), 0.0, dtype=torch.float64)
        self.num_steps_done.share_memory_()
        self.time_taken.share_memory_()
        self.report_queue = report_queue
        super().__init__(
            mp_ctx,
            ReportWorkerProcess,
            port,
            config,
            report_queue,
            my_t_zero,
            self.num_steps_done,
            self.time_taken,
        )

        self.response_queue.get()

    def start_collection(self):
        self.report_queue.put(
            (ReportWorkerTasks.start_collection, time.perf_counter())
        )

    def state_dict(self):
        self.report_queue.put((ReportWorkerTasks.state_dict, None))
        return self.response_queue.get()

    def load_state_dict(self, state_dict):
        if state_dict is not None:
            self.report_queue.put(
                (ReportWorkerTasks.load_state_dict, state_dict)
            )
