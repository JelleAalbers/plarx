from collections import defaultdict
import concurrent.futures as cf
from enum import Enum
import os
from random import random
import time
import typing as ty

import psutil

import plarx
from .common import exporter

export, __all__ = exporter()


class Signals(Enum):
    """From _get_new_task to main_loop"""
    NO_TASK = 0
    GOT_TASK = 1
    WAIT_EXTERNAL = 3


@export
class Stream:
    pending_tasks: ty.List[plarx.Task]
    stored_data: ty.Dict[str, plarx.Buffer]  # {dtypename: StoredData}
    final_target: str
    jobs: ty.List[plarx.Job]
    this_process: psutil.Process
    threshold_mb = 1000

    def __init__(self, jobs: ty.List[plarx.Job],
                 yield_outputs=None, max_workers=5):
        if isinstance(yield_outputs, str):
            yield_outputs = [yield_outputs]

        self.max_workers = max_workers
        self.jobs = jobs

        def get_priority(tg: plarx.Job):
            return tg.priority, tg.depth
        self.jobs.sort(key=get_priority)

        self.pending_tasks = []
        self.this_process = psutil.Process(os.getpid())

        self.last_yielded_chunk = -1

        self.executors = dict(
            process=cf.ProcessPoolExecutor(max_workers=self.max_workers),
            thread=cf.ThreadPoolExecutor(max_workers=self.max_workers))
        # TODO: how to pass to jobs?
        for job in jobs:
            job.executors = self.executors

        who_wants = defaultdict(list)
        for job in self.jobs:
            if not len(job.depends_on) and not job.is_source:
                raise RuntimeError(
                    f"{job} has no dependencies but it is not a source?")
            for dt in job.depends_on:
                who_wants[dt].append(job)
        for dt in yield_outputs:
            who_wants.setdefault(dt, [])

        self.stored_data = {
            dt: plarx.Buffer(dt, jobs, yield_to_user=dt in yield_outputs)
            for dt, jobs in who_wants.items()}

    def main_loop(self):
        while True:
            self._receive_from_done_tasks()

            for sd in self.stored_data.values():
                yield from sd.yield_to_user()
                sd.cleanup()

            status, task = self._get_new_task()

            if status is Signals.GOT_TASK:
                self.pending_tasks += [task]
                if len(self.pending_tasks) < self.max_workers:
                    continue
                self.wait_until_task_done()

            elif status is Signals.NO_TASK:
                if not self.pending_tasks:
                    self._exit_normally()
                    return
                self.wait_until_task_done()
                continue

            elif status is Signals.WAIT_EXTERNAL:
                # TODO: emit who is waiting
                self._emit_status(f"Waiting on external condition")
                time.sleep(5)
                continue

    def _exit_normally(self):
        if any([not job.all_results_arrived
                for job in self.jobs]):
            self.exit_with_exception(RuntimeError(
                "No available or pending tasks, "
                "but data is not exhausted!"))
        if any([sd.n_stored() for sd in self.stored_data.values()]):
            self.exit_with_exception(RuntimeError(
                "End but data is still stored!"))

        for exc in self.executors.values():
            exc.shutdown()

    def wait_until_task_done(self):
        while True:
            done, not_done = cf.wait(
                [t.future for t in self.pending_tasks],
                return_when=cf.FIRST_COMPLETED,
                timeout=5)
            if len(done):
                break
            self._emit_status("Waiting for a task to complete")

    def _receive_from_done_tasks(self):
        still_pending = []
        for task in self.pending_tasks:
            f = task.future
            if not f.done():
                still_pending.append(task)
                continue

            # Get the result
            try:
                result = task.job.get_result(task)
            except Exception as e:
                self.exit_with_exception(e, f"Exception from {task}")
                raise RuntimeError("Exit failed??")  # Pycharm likes

            if result is None:
                continue

            for dtype, result in result.items():
                if dtype not in self.stored_data:
                    # TODO: make configurable error
                    # print(f"Got {dtype} which nobody wants, discarding...")
                    continue
                self.stored_data[dtype].add(data=result, chunk_i=task.chunk_i)

        self.pending_tasks = still_pending

    def _get_new_task(self):
        """Return a new task, or None, if it is wiser or necessary to wait"""
        external_waits = []    # Jobs waiting for external conditions
        sources = []           # Sources we could load more data from
        requests_for = defaultdict(int)  # Requests for particular inputs
        exhausted_dtypes = self._get_exhausted()

        for job in self.jobs:
            if not job.could_submit_new_task():
                continue

            # Are any of the required inputs exhausted?
            exhausted = False
            if job.is_source:
                exhausted = job.external_inputs_exhausted()
            else:
                for dt in job.wants_input:
                    if dt in exhausted_dtypes:
                        exhausted = True

            # If so, start a cleanup task
            if exhausted:
                if job.final_task_submitted:
                    continue
                inputs = {dt: self.stored_data[dt].slurp_for(job)
                          for dt in job.depends_on}
                return Signals.GOT_TASK, job.get_cleanup_task(inputs)

            # Check external conditions satisfied
            if (not job.external_input_ready()
                    and not job.external_inputs_exhausted()):
                external_waits.append(job)
                continue

            if job.is_source:
                # Handle these separately (at the end) regardless of priority
                sources.append(job)
                continue

            # Any input missing?
            missing_input = False
            for dtype, chunk_id in job.wants_input.items():
                if not self.stored_data[dtype].has_stored(chunk_id):
                    requests_for[dtype] += 1
                    missing_input = True
                    break
            if missing_input:
                continue

            # We're good! Grab all inputs and submit.
            task_inputs = dict()
            for dtype, chunk_i in job.wants_input.items():
                task_inputs[dtype] = self.stored_data[dtype].grab_for(
                    job=job, chunk_i=chunk_i)

            return Signals.GOT_TASK, job.get_task(task_inputs)

        if sources:
            # No computation tasks to do, but we could load new data
            if (self.this_process.memory_info().rss / 1e6 > self.threshold_mb
                    and self.pending_tasks):
                # ... Let's not though; instead wait for current tasks.
                # (We could perhaps also wait for an external condition
                # but in all likelihood a task will complete soon enough)
                return Signals.NO_TASK, None
            # Load data for the source that is blocking the most tasks
            # Jitter it a bit for better performance on ties..
            # TODO: There is a better way, but I'm too lazy now
            requests_for_source = [(s, sum([requests_for.get(dt, 0)
                                            for dt in s.provides]) + random())
                                   for s in sources]
            s, _ = max(requests_for_source, key=lambda q: q[1])
            return Signals.GOT_TASK, s.get_task(None)

        if external_waits:
            if len(self.pending_tasks):
                # Assume an existing task will complete before the wait time.
                # TODO: Good idea? Maybe make configurable?
                return Signals.NO_TASK, None
            else:
                return Signals.WAIT_EXTERNAL, None

        # No work to do. Maybe a pending task will still generate some though.
        return Signals.NO_TASK, None

    def _emit_status(self, msg):
        print(msg)
        print(f"\tPending tasks: {self.pending_tasks}")

    def exit_with_exception(self, exception, extra_message=''):
        print(extra_message)
        for job in self.jobs:
            print(f"{job}:"
                  f"\n\twants {job.wants_input}, "
                  f"\n\tlast_submitted {job.last_submitted_i}, "
                  f"\n\thighest_cont._done {job.highest_continuous_done_i}, "
                  f"\n\tfinal_task_submitted {job.final_task_submitted},"
                  f"\n\tall_results_arrived {job.all_results_arrived},"
                  f"\n\tis source {job.is_source}")
        for dt, d in self.stored_data.items():
            d.print_status()
        for job in self.jobs:
            if not job.final_task_submitted:
                try:
                    job.cleanup(exception=exception, chunk_i=-1)
                except Exception as e:
                    print(f"Exceptional shutdown of {job} failed")
                    print(f"Got another exception: {e}")
                    pass   # These are exceptional times...
            raise exception
        for t in self.pending_tasks:
            t.future.cancel()
        self.threadpool.shutdown()
        self.processpool.shutdown()

    def _get_exhausted(self):
        """Return list of provided datatypes that are exhausted"""
        # Note final tasks can return results, so we check for
        # all_results_arrived, not final_task_submitted!
        exhausted = []
        for job in self.jobs:
            if not job.provides or not job.all_results_arrived:
                continue
            for p in job.provides:
                if not self.stored_data[p].n_stored():
                    exhausted.append(p)
        return exhausted
