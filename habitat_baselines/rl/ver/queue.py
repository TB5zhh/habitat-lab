import importlib
from typing import TYPE_CHECKING

use_faster_fifo = importlib.util.find_spec("faster_fifo") is not None


if not TYPE_CHECKING and use_faster_fifo:
    import faster_fifo
    import faster_fifo_reduction  # noqa: F401

    BatchedQueue = faster_fifo.Queue
else:
    import multiprocessing.queues
    import queue
    import time
    import warnings

    warnings.warn(
        "Unable to import faster_fifo."
        " Using the fallback. This may reduce performance."
    )

    class BatchedQueue(multiprocessing.queues.Queue):
        def get_many(
            self,
            block=True,
            timeout=10.0,
            max_messages_to_get=1_000_000_000,
        ):
            msgs = [self.get(block, timeout)]
            while len(msgs) < max_messages_to_get:
                try:
                    msgs.append(self.get_nowait())
                except queue.Empty:
                    break

            return msgs

        def put_many(self, xs, block=True, timeout=10.0):

            t_start = time.perf_counter()
            n_put = 0
            for x in xs:
                self.put(x, block, timeout - (t_start - time.perf_counter()))
                n_put += 1

            if n_put != len(xs):
                raise RuntimeError(
                    f"Couldn't put all. Put {n_put}, needed to put {len(xs)}"
                )
