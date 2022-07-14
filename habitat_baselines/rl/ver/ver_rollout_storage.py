import functools
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    _np_invert_permutation,
    build_pack_info_from_episode_ids,
    build_rnn_build_seq_info,
)


def swap_inds_slice(src_inds: torch.Tensor, dst_slice: slice, t: torch.Tensor):
    assert len(src_inds) == (
        dst_slice.stop - (dst_slice.start or 0)
    ), f"{src_inds} {dst_slice}"
    if len(src_inds) == 0:
        return

    src = t[src_inds]
    dst = t[dst_slice].clone()
    t[src_inds] = dst
    t[dst_slice] = src


def partition_n_into_p(n: int, p: int) -> List[int]:
    r"""Creates a partitioning of n elements into p bins."""
    return [n // p + (1 if i < (n % p) else 0) for i in range(p)]


def generate_ragged_mini_batches(
    num_mini_batch: int,
    sequence_lengths: np.ndarray,
    num_seqs_at_step: np.ndarray,
    select_inds: np.ndarray,
    last_sequence_in_batch_mask: np.ndarray,
    episode_ids: np.ndarray,
    include_all: bool = False,
) -> Iterator[np.ndarray]:
    sequence_lengths = sequence_lengths.copy()
    if not include_all:
        sequence_lengths[last_sequence_in_batch_mask] -= 1

    mb_sizes = np.array(
        partition_n_into_p(int(np.sum(sequence_lengths)), num_mini_batch),
        dtype=np.int64,
    )
    # Exclusive cumsum
    offset_to_step = np.cumsum(num_seqs_at_step) - num_seqs_at_step

    mb_inds = np.empty((num_mini_batch, mb_sizes.max()), dtype=np.int64)
    seq_ordering = np.random.permutation(len(sequence_lengths))
    seq_ptr = 0
    for mb_idx in np.random.permutation(num_mini_batch):
        next_seq = seq_ordering[seq_ptr]
        mb_size = mb_sizes[mb_idx]
        ptr = 0
        while ptr < mb_size:
            while sequence_lengths[next_seq] == 0:
                seq_ptr += 1
                assert seq_ptr < len(seq_ordering)
                next_seq = seq_ordering[seq_ptr]

            len_to_use = min(sequence_lengths[next_seq], mb_size - ptr)

            offset = sequence_lengths[next_seq] - len_to_use

            mb_inds[mb_idx, ptr : ptr + len_to_use] = select_inds[
                next_seq + offset_to_step[offset : offset + len_to_use]
            ]
            # Sanity check
            assert (
                len(
                    np.unique(
                        episode_ids[mb_inds[mb_idx, ptr : ptr + len_to_use]]
                    )
                )
                == 1
            ), episode_ids[mb_inds[mb_idx, ptr : ptr + len_to_use]]

            ptr += len_to_use

            sequence_lengths[next_seq] = offset

        assert ptr == mb_size

    assert np.sum(sequence_lengths) == 0

    for mb_idx in np.random.permutation(num_mini_batch):
        yield mb_inds[mb_idx, 0 : mb_sizes[mb_idx]]


class VERRolloutStorage(RolloutStorage):
    ptr: np.ndarray
    prev_inds: np.ndarray
    num_steps_collected: np.ndarray
    rollout_done: np.ndarray
    cpu_current_policy_version: np.ndarray
    actor_steps_collected: np.ndarray
    current_steps: np.ndarray
    will_replay_step: np.ndarray
    _first_rollout: np.ndarray

    next_hidden_states: torch.Tensor
    next_prev_actions: torch.Tensor
    current_policy_version: torch.Tensor

    def __init__(
        self,
        variable_experience: bool,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            action_shape,
            is_double_buffered,
            discrete_actions,
        )
        self.use_is_coeffs = variable_experience

        if self.use_is_coeffs:
            self.buffers["is_coeffs"] = torch.ones_like(
                self.buffers["returns"]
            )

        for k in (
            "policy_version",
            "environment_ids",
            "episode_ids",
            "step_ids",
        ):
            self.buffers[k] = torch.zeros_like(
                self.buffers["returns"], dtype=torch.int64
            )

        self.buffers["is_stale"] = torch.ones_like(
            self.buffers["returns"], dtype=torch.bool
        )

        self.variable_experience = variable_experience
        self.buffer_size = (self.num_steps + 1) * self._num_envs

        self._aux_buffers = TensorDict()
        self.aux_buffers_on_device = []

        self._aux_buffers["next_hidden_states"] = self.buffers[
            "recurrent_hidden_states"
        ][0].clone()
        self._aux_buffers["next_prev_actions"] = self.buffers["prev_actions"][
            0
        ].clone()
        self._aux_buffers["current_policy_version"] = torch.ones(
            (1, 1), dtype=torch.int64
        )

        self.aux_buffers_on_device += list(self._aux_buffers.keys())

        self._aux_buffers["cpu_current_policy_version"] = torch.ones(
            (1, 1), dtype=torch.int64
        )

        self._aux_buffers["num_steps_collected"] = torch.zeros(
            (1,), dtype=torch.int64
        )
        self._aux_buffers["rollout_done"] = torch.zeros((1,), dtype=torch.bool)
        self._aux_buffers["current_steps"] = torch.zeros(
            (num_envs,), dtype=torch.int64
        )
        self._aux_buffers["actor_steps_collected"] = self._aux_buffers[
            "current_steps"
        ].clone()

        self._aux_buffers["ptr"] = torch.zeros((1,), dtype=torch.int64)
        self._aux_buffers["prev_inds"] = torch.full(
            (num_envs,), -1, dtype=torch.int64
        )
        self._aux_buffers["_first_rollout"] = torch.full(
            (1,), True, dtype=torch.bool
        )
        self._aux_buffers["will_replay_step"] = torch.zeros(
            (num_envs,), dtype=torch.bool
        )

        if self.variable_experience:
            self.buffers.map_in_place(lambda t: t.flatten(0, 1))

        self.aux_buffers_on_device = set(self.aux_buffers_on_device)
        self._set_aux_buffers()

    @property
    def num_steps_to_collect(self) -> int:
        if self._first_rollout:
            return self.buffer_size
        else:
            return self._num_envs * self.num_steps

    def _set_aux_buffers(self):
        for k, v in self.__annotations__.items():
            if k not in self._aux_buffers:
                if v in (torch.Tensor, np.ndarray):
                    raise RuntimeError(f"Annotation {k} not in aux buffers")
                else:
                    continue

            if k in self.aux_buffers_on_device:  # noqa: SIM401
                buf = self._aux_buffers[k]
            else:
                buf = self._aux_buffers[k].numpy()

            assert isinstance(
                buf, v
            ), f"Expected aux buffer of type {v} but got {type(buf)}"
            setattr(self, k, buf)

    def __getstate__(self):
        state = super().__getstate__().copy()
        for k in self.__annotations__.items():
            if k in self._aux_buffers:
                del state[k]

        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        self._set_aux_buffers()

    def copy(self, other: "VERRolloutStorage"):
        self.buffers[:] = other.buffers
        self._aux_buffers[:] = other._aux_buffers

    def share_memory_(self):
        self.buffers.map_in_place(lambda t: t.share_memory_())

        self._aux_buffers.map_in_place(lambda t: t.share_memory_())

        self._set_aux_buffers()

    def to(self, device):
        super().to(device)

        self._aux_buffers.map_in_place(
            lambda t, k: t.to(device=device)
            if k in self.aux_buffers_on_device
            else t
        )

        self._set_aux_buffers()

    def after_update(self):
        self.current_steps[:] = 1
        self.current_steps[self.will_replay_step] -= 1
        self.buffers["is_stale"].fill_(True)

        if not self.variable_experience:
            assert np.all(self.will_replay_step)
            self.next_hidden_states[:] = self.buffers[
                "recurrent_hidden_states"
            ][-1]
            self.next_prev_actions[:] = self.buffers["prev_actions"][-1]
        else:
            # With ver, we will have some actions
            # in flight as we can't reset the simulator. In that case we need
            # to save the prev rollout step data for that action (as the
            # reward for the action in flight is for the prev rollout
            # step). We put those into [0:num_with_action_in_flight]
            # because that range won't be overwritten.
            has_action_in_flight = np.logical_not(self.will_replay_step)
            num_with_action_in_flight = np.count_nonzero(has_action_in_flight)

            self.buffers.apply(
                functools.partial(
                    swap_inds_slice,
                    self.prev_inds[has_action_in_flight],
                    slice(num_with_action_in_flight),
                )
            )

            # For previous steps without an action in flight,
            # we will "replay" that step of experience
            # so we can use the new policy for inference.
            # We need to make sure they get overwritten in the next
            # rollout, so we place them at the
            # start of where we will write during the next rollout.
            self.buffers.apply(
                functools.partial(
                    swap_inds_slice,
                    self.prev_inds[np.logical_not(has_action_in_flight)],
                    slice(num_with_action_in_flight, self._num_envs),
                )
            )

            self.prev_inds[:] = -1
            self.prev_inds[has_action_in_flight] = np.arange(
                num_with_action_in_flight,
                dtype=self.prev_inds.dtype,
            )
            self.will_replay_step[:] = False
            self.ptr[:] = num_with_action_in_flight

            # For the remaining steps, we order them such that the oldest experience
            # get's overwritten first then we order the experience
            # so that the actor with the most experience collected
            # gets overwritten first

            environment_ids = self.buffers["environment_ids"].view(-1)[
                self._num_envs :
            ]
            unique_environment_ids = torch.arange(
                self._num_envs,
                dtype=environment_ids.dtype,
                device=environment_ids.device,
            )
            actor_mask = environment_ids.view(
                -1, 1
            ) == unique_environment_ids.view(1, -1)
            rollout_step_inds = torch.masked_select(
                torch.cumsum(actor_mask, 0, dtype=torch.int64),
                actor_mask,
            )

            actor_max_step = torch.zeros_like(actor_mask, dtype=torch.int64)
            actor_max_step.masked_scatter_(actor_mask, rollout_step_inds)
            actor_max_step, _ = actor_max_step.max(0)

            sort_keys = (
                torch.masked_select(actor_max_step.view(1, -1), actor_mask)
                - rollout_step_inds
            ) * self._num_envs + environment_ids
            #  assert torch.unique(sort_keys).numel() == sort_keys.numel()

            _, ordering = torch.sort(sort_keys, descending=True)

            # Do a second ordering so that experience from older policies
            # is definitely first to be overwritten.
            _, version_ordering = torch.sort(
                self.current_policy_version.view(-1)
                - (
                    self.buffers["policy_version"]
                    .view(-1)[self._num_envs :]
                    .index_select(0, ordering)
                ),
                descending=True,
                stable=True,
            )
            combined_ordering = ordering.index_select(0, version_ordering)

            self.buffers.apply(
                lambda t: t[self._num_envs :].copy_(
                    t[self._num_envs :].index_select(0, combined_ordering)
                )
            )

        self.num_steps_collected[:] = 0
        self.rollout_done[:] = False
        self._first_rollout[:] = False

    def increment_policy_version(self):
        self.current_policy_version += 1
        self.cpu_current_policy_version += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def after_rollout(self):
        self.buffers["is_stale"][:] = (
            self.buffers["policy_version"] < self.current_policy_version
        )

        self.current_rollout_step_idxs[0] = self.num_steps + 1

        if self.use_is_coeffs:
            environment_ids = self.buffers["environment_ids"].view(-1)
            unique_actors = torch.unique(environment_ids, sorted=False)
            samples_per_actor = (
                (environment_ids.view(1, -1) == unique_actors.view(-1, 1))
                .float()
                .sum(-1)
            )
            is_per_actor = torch.empty(
                (self._num_envs,), dtype=torch.float32, device=self.device
            )
            # Use a scatter so that is_per_actor.size() == num_envs
            # and so that we don't have to have unique_actors be sorted
            is_per_actor.scatter_(
                0,
                unique_actors.view(-1),
                # With uniform sampling, we'd get (numsteps + 1)
                # per actor
                (self.num_steps + 1) / samples_per_actor,
            )
            self.buffers["is_coeffs"].copy_(
                is_per_actor[environment_ids].view(-1, 1)
            )

    def compute_returns(
        self,
        use_gae,
        gamma,
        tau,
    ):
        if not use_gae:
            tau = 1.0

        not_masks = torch.logical_not(self.buffers["masks"]).to(
            device="cpu", non_blocking=True
        )
        self.episode_ids_cpu = self.buffers["episode_ids"].to(
            device="cpu", non_blocking=True
        )
        self.environment_ids_cpu = self.buffers["environment_ids"].to(
            device="cpu", non_blocking=True
        )
        self.step_ids_cpu = self.buffers["step_ids"].to(
            device="cpu", non_blocking=True
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.dones_cpu = not_masks.view(-1, self._num_envs).numpy()
        self.episode_ids_cpu = self.episode_ids_cpu.view(-1).numpy()
        self.environment_ids_cpu = self.environment_ids_cpu.view(-1).numpy()
        self.step_ids_cpu = self.step_ids_cpu.view(-1).numpy()

        rewards = self.buffers["rewards"].to(device="cpu", non_blocking=True)
        returns = self.buffers["returns"].to(device="cpu", non_blocking=True)
        is_not_stale = torch.logical_not(self.buffers["is_stale"]).to(
            device="cpu", non_blocking=True
        )
        values = self.buffers["value_preds"].to(
            device="cpu", non_blocking=True
        )

        rnn_build_seq_info = build_pack_info_from_episode_ids(
            self.episode_ids_cpu,
            self.environment_ids_cpu,
            self.step_ids_cpu,
        )

        (
            self.select_inds,
            self.num_seqs_at_step,
            self.sequence_lengths,
            self.sequence_starts,
            self.last_sequence_in_batch_mask,
        ) = (
            rnn_build_seq_info["select_inds"],
            rnn_build_seq_info["num_seqs_at_step"],
            rnn_build_seq_info["sequence_lengths"],
            rnn_build_seq_info["sequence_starts"],
            rnn_build_seq_info["last_sequence_in_batch_mask"],
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        rewards, values, is_not_stale = map(
            lambda t: t.view(-1, 1).numpy()[self.select_inds],
            (rewards, values, is_not_stale),
        )
        returns = returns.view(-1, 1).numpy()
        returns[:] = returns[self.select_inds]

        gae = np.zeros((self.num_seqs_at_step[0], 1))
        last_values = gae.copy()
        ptr = returns.size
        for len_minus_1, n_seqs in reversed(
            list(enumerate(self.num_seqs_at_step))
        ):
            curr_slice = slice(ptr - n_seqs, ptr)
            q_est = rewards[curr_slice] + gamma * last_values[:n_seqs]
            delta = q_est - values[curr_slice]
            gae[:n_seqs] = delta + (tau * gamma) * gae[:n_seqs]

            is_last_step = self.sequence_lengths == (len_minus_1 + 1)
            is_invalid = is_last_step & self.last_sequence_in_batch_mask

            # For the last episode from each worker, we do an extra loop
            # to fill last_values with the bootstrap.  So we re-zero
            # the GAE value here
            gae[is_invalid] = 0.0

            # If the step isn't stale or we don't have a return
            # calculate, use the newly calculated return value,
            # otherwise keep the current one
            use_new_value = is_not_stale[curr_slice] | np.logical_not(
                np.isfinite(returns[curr_slice])
            )
            returns[curr_slice][use_new_value] = (
                gae[:n_seqs] + values[curr_slice]
            )[use_new_value]

            # Mark anything that's invalid with nan
            returns[curr_slice][is_invalid[:n_seqs]] = float("nan")

            last_values[:n_seqs] = values[curr_slice]

            ptr -= n_seqs

        assert ptr == 0

        returns[:] = returns[_np_invert_permutation(self.select_inds)]
        returns = torch.from_numpy(returns).view_as(self.buffers["returns"])

        if not self.variable_experience:
            assert torch.all(torch.isfinite(returns[:-1])), dict(
                returns=returns.squeeze(),
                dones=self.dones_cpu,
                episode_ids=self.episode_ids_cpu.reshape(-1, self._num_envs),
                environment_ids=self.environment_ids_cpu.reshape(
                    -1, self._num_envs
                ),
                step_ids=self.step_ids_cpu.reshape(-1, self._num_envs),
                is_not_stale=is_not_stale[
                    _np_invert_permutation(self.select_inds)
                ].reshape(-1, self._num_envs),
            )
        else:
            assert torch.count_nonzero(torch.isfinite(returns)) == (
                self.num_steps * self._num_envs
            ), returns.squeeze().numpy()[self.select_inds]

        self.buffers["returns"].copy_(returns, non_blocking=True)
        self.current_rollout_step_idxs[0] = self.num_steps

    def recurrent_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
        include_all: bool = False,
    ) -> Iterator[TensorDict]:
        force_ragged_generation = False
        if not force_ragged_generation and not self.variable_experience:
            yield from super().recurrent_generator(
                advantages, num_mini_batch, include_all
            )
        else:
            for mb_inds in generate_ragged_mini_batches(
                num_mini_batch,
                self.sequence_lengths,
                self.num_seqs_at_step,
                self.select_inds,
                self.last_sequence_in_batch_mask,
                self.episode_ids_cpu,
                include_all,
            ):
                mb_inds_cpu = torch.from_numpy(mb_inds)
                mb_inds = mb_inds_cpu.to(device=self.device)

                if not self.variable_experience:
                    batch = self.buffers.map(lambda t: t.flatten(0, 1))[
                        mb_inds
                    ]
                    if advantages is not None:
                        batch["advantages"] = advantages.flatten(0, 1)[mb_inds]
                else:
                    batch = self.buffers[mb_inds]
                    if advantages is not None:
                        batch["advantages"] = advantages[mb_inds]

                batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                    device=self.device,
                    build_fn=lambda: build_pack_info_from_episode_ids(
                        self.episode_ids_cpu[mb_inds_cpu],
                        self.environment_ids_cpu[mb_inds_cpu],
                        self.step_ids_cpu[mb_inds_cpu],
                    ),
                )

                rnn_build_seq_info = batch["rnn_build_seq_info"]
                batch["recurrent_hidden_states"] = batch[
                    "recurrent_hidden_states"
                ].index_select(
                    0,
                    rnn_build_seq_info["sequence_starts"][
                        rnn_build_seq_info["first_sequence_in_batch_mask"]
                    ],
                )

                yield batch
