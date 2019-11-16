from collections import defaultdict
import concurrent.futures as cf
from enum import Enum
from functools import partial
import os
from random import random
import time
import typing as ty
from types import MethodType

import psutil
import numpy as np

from .utils import exporter, random_str
export, __all__ = exporter()


class Signals(Enum):
    NO_TASK = 0
    GOT_TASK = 1
    WAIT_EXTERNAL = 3


class Task(ty.NamedTuple):
    """Task object, tracking a future submitted to a pool"""

    # If true, mark Job as finished on completion
    is_final: bool

    # Generator that made the task.
    generator: ty.Any       # Wait for python 4.0 / future annotations

    # Chunk number about to be produced
    chunk_i: int

    # Future object. Result will be {dtypename: array, ...}
    # or possibly, and only for the final task, None.
    #
    # For tasks that are placeholders for data to be returned to the user,
    # future directly stores the result.
    future: ty.Union[cf.Future, np.ndarray]

    def __repr__(self):
        return f"{self.generator}:{self.chunk_i}"


class CannotSubmitNewTask(Exception):
    pass


@export
class Job:
    provides: ty.Tuple[str] = tuple()     # Produced data types
    depends_on: ty.Tuple[str] = tuple()   # Input data types

    submit_to = 'thread'        # thread, process, or user
    parallel = True             # Can we start more than one task at a time?
    changing_inputs = False     # If True, task_function returns
                                # (result, inputs_wanted_next_time)
                                # Cannot parallelize.
    priority = 0                # 0 = saver/target, 1 = source, 2 = other

    wants_input: ty.Dict[str, int]  # [(dtype, chunk_i), ...] of inputs
                                    # needed to make progress
    seen_input: ty.Dict[str, int]   # [(dtype, chunk_i), ...] of inputs
                                    # already seen

    depth = 0                   # Dependency hops to final target

    pending_is: set             # Set of pending task numbers
    last_submitted_i = -1       # Number of last emitted task
    _highest_done_i = -1        # Highest completed task number, NOT continuous!
    final_task_i = float('inf')   # Number of final task (if known)

    # "Finished" variables. Note there are two...
    final_task_submitted = False
    all_results_arrived = False

    @property
    def highest_continuous_done_i(self):
        if self.pending_is:
            return min(self.pending_is) - 1
        else:
            return self._highest_done_i

    @property
    def is_source(self):
        return not len(self.depends_on)

    # TODO: how to keep defaults consistent here?
    @classmethod
    def from_function(
            cls,
            task: ty.Callable,
            cleanup: ty.Callable=None,
            provides=tuple(),
            depends_on=tuple(),
            submit_to='thread',
            parallel=True,
            changing_inputs=False):
        """Construct job from task function
        :param task: Task function. See Job.task
        :param cleanup: Cleanup function. See Job.cleanup

        Other arguments are as in first lines of Job code (TODO:document)
        """
        if not len(provides) and len(depends_on):
            raise ValueError("Job must provide or depend on something")
        self = type('Job' + random_str(10),
                    (Job,),
                    dict(provides=provides, depends_on=depends_on,
                        submit_to=submit_to, parallel=parallel,
                        changing_inputs=changing_inputs))
        self.task = MethodType(task, self)
        if self.cleanup is not None:
            self.cleanup = MethodType(cleanup, self)

    def __init__(self):
        if self.changing_inputs and self.parallel:
            raise RuntimeError("Cannot parallelize a Job with "
                               "indirect input delivery.")
        if self.is_source and len(self.depends_on):
            raise RuntimeError("A source cannot depend on any data type")
        self.wants_input = {dt: 0 for dt in self.depends_on}
        self.seen_input = {dt: -1 for dt in self.depends_on}
        self.pending_is = set()

    def __repr__(self):
        _from = ','.join(self.depends_on)
        _to = ','.join(self.provides)
        return f"[{_from}>{_to}]"

    def _prepare_get_task(self, inputs):
        self._check_can_submit()

        task_i = self.last_submitted_i + 1
        self.pending_is.add(task_i)

        if self.is_source:
            assert inputs is None or not len(inputs), "Passed inputs to source"
            inputs = dict()
        return task_i, inputs

    def get_task(self, inputs=None):
        """Submit a task to operate on inputs"""
        task_i, inputs = self._prepare_get_task(inputs)

        # Validate inputs
        # Inputs must be dicts of numpy arrays
        for k in inputs:
            if not isinstance(inputs[k], np.ndarray):
                raise RuntimeError(f"Got {type(inputs[k])} instead of np"
                                   f"array as input {k} given to {self}")
            if k not in self.wants_input:
                raise RuntimeError(f"Unwanted input {k} given to {self}")
        for k in self.wants_input:
            if k not in inputs:
                raise RuntimeError(f"Missing input {k} to {self}")

        for k in inputs:
            self.seen_input[k] += 1
        content = partial(self.task, chunk_i=task_i, **inputs)

        if not self.changing_inputs:
            # Request the next set of inputs
            for dt in self.wants_input:
                self.wants_input[dt] += 1

        return self._submit_task(content, task_i, is_final=False)

    def _submit_task(self, content, task_i, is_final) -> (Signals, Task):
        future = self.executors[self.submit_to].submit(content)
        self.last_submitted_i += 1
        return Task(
            future=future,
            generator=self,
            chunk_i=task_i,
            is_final=is_final)

    def get_cleanup_task(self, inputs=None):
        task_i, inputs = self._prepare_get_task(inputs)
        # TODO: input lists of dicts of unprovided inputs, check it

        self.final_task_submitted = True
        self.final_task_i = task_i
        content = partial(self.cleanup, chunk_i=task_i, **inputs)
        return self._submit_task(content, task_i, is_final=True)

    def could_submit_new_task(self):
        """Return if we are ready to submit a new task
        provided the required inputs are available.
        """
        try:
            self._check_can_submit()
            return True
        except CannotSubmitNewTask:
            return False

    def _check_can_submit(self):
        """Raises RuntimeError unless we are ready to submit a new task"""
        if not self.parallel and self.last_submitted_i != self.highest_continuous_done_i:
            raise CannotSubmitNewTask(
                f"Attempt to get task for {self} "
                f"out of order: last submitted {self.last_submitted_i}, "
                f"but highest_continuous_done_i is {self.highest_continuous_done_i}.")
        if self.all_results_arrived:
            raise CannotSubmitNewTask(
                f"Can't get {self} task: all results already arrived")
        if self.final_task_submitted:
            raise CannotSubmitNewTask(
                "Can't get {self} task: final task already submitted")

    def get_result(self, task: Task):
        """Return result of task"""
        # Basic bookkeeping
        assert task.chunk_i in self.pending_is
        self.pending_is.discard(task.chunk_i)
        self._highest_done_i = max(self._highest_done_i, task.chunk_i)
        if self.highest_continuous_done_i == self.final_task_i:
            self.all_results_arrived = True

        # Fetch result
        if not task.future.done():
            raise RuntimeError("get_result called before task was done")
        result = task.future.result()  # Will raise if exception

        # Record new inputs
        if self.changing_inputs:
            if task.is_final:
                new_inputs = {}
            else:
                assert isinstance(result, tuple) and len(result) == 2, \
                    f"{self} changes inputs but returned a {type(result)} " \
                    f"rather than a two-tuple"
                result, new_inputs = result
            self.wants_input = {dt: self.seen_input[dt] + 1
                                for dt in new_inputs}

        # Check and return result
        if result is None:
            assert task.is_final, f"{task} is not final but returned None"
        else:
            self._validate_results(task, result)
        return result

    def _validate_results(self, task, result):
        """Check if task returned the correct result"""
        assert isinstance(result, dict), \
            f"{task} returned a {type(result)} rather than a dict"
        for k in result:
            assert k in self.provides, \
                f"{task} provided unwanted output {k}"
        for k in self.provides:
            assert k in result, \
                f"{task} failed to provide needed output {k}"

    ##
    # Functions to override
    ##

    def task(self, chunk_i: int, **kwargs) \
            -> ty.Dict[str, np.ndarray]:
        raise NotImplementedError

    def cleanup(self, chunk_i: int, exception=None, **inputs) \
            -> ty.Union[None, ty.Dict[str, np.ndarray]]:
        """Execute final cleanup, e.g. closing of files.

        :param chunk_i: Task number, or -1 in exceptional termination.
        :param exception: Exception object that caused the processing to crash,
        or None during normal termination.
        :param **inputs: During normal termination, dictionary of
        lists of inputs that were not passed to task_function yet.
        During exceptional termination, this is None.

        Optionally, return a final result, just like task_function.
        """
        pass

    def external_inputs_exhausted(self):
        """For sources, return whether external inputs exhausted"""
        return False

    def external_input_ready(self):
        """For sources, return whether the next external input
         (i.e. self.chunk_id + 1) is ready"""
        return True


class StoredData:
    dtype: str                  # Name of data type
    stored: ty.Dict[int, np.ndarray]
                                # [(chunk_i, data), ...]
    seen_by_consumers: ty.Dict[Job, int]
                                # Last chunk seen by each of the tasks
                                # that need it
    last_contiguous = -1        # Latest chunk that has arrived
                                # for which all previous chunks have arrived

    yielding_to_user = False       # Output must be yielded to the user
    last_yielded_to_user = -1    # Last chunk yielded to the user

    def __init__(self,
                 dtype,
                 wanted_by: ty.List[Job],
                 yield_to_user=False, ):
        self.dtype = dtype
        self.stored = dict()
        self.seen_by_consumers = {tg: -1
                                  for tg in wanted_by}
        self.yielding_to_user = yield_to_user

    def add(self, data: np.ndarray, chunk_i: int):
        self.stored[chunk_i] = data
        if self.last_contiguous == chunk_i - 1:
            self.last_contiguous += 1

    def grab_for(self, chunk_i, tg) -> np.ndarray:
        assert self.seen_by_consumers[tg] < chunk_i
        self.seen_by_consumers[tg] = chunk_i
        return self.stored[chunk_i]

    def slurp_for(self, tg) -> ty.List[np.ndarray]:
        result = []
        while self.seen_by_consumers[tg] < self.last_contiguous:
            result.append(self.grab_for(
                tg=tg,
                chunk_i=self.seen_by_consumers[tg] + 1))
        return result

    def yield_to_user(self):
        if not self.yielding_to_user:
            return

        chunk_i = self.last_yielded_to_user + 1
        while chunk_i in self.stored:
            result = self.stored[chunk_i]
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"Attempt to yield a {type(result)} rather "
                    f"than a numpy array to the user")
            yield result  #, self.dtype, chunk_i
            self.last_yielded_to_user = chunk_i
            chunk_i += 1

    def cleanup(self):
        seen_by_all = min(self.seen_by_consumers.values(),
                          default=float('inf'))
        if self.yielding_to_user:
            seen_by_all = min(seen_by_all, self.last_yielded_to_user)
        elif not len(self.seen_by_consumers):
            raise RuntimeError(f"{self.dtype} is not consumed by anyone??")

        self.stored = {chunk_i: data
                       for chunk_i, data in self.stored.items()
                       if chunk_i > seen_by_all}

    def print_status(self):
        print(f"{self.dtype}: stored: {list(self.stored.keys())}, "
              f"last_contiguous: {self.last_contiguous}, "
              f"seen: {self.seen_by_consumers}")

    def has_stored(self, chunk_id):
        return chunk_id in self.stored

    def n_stored(self):
        return len(self.stored)

    def __repr__(self):
        return f'StoredData[{self.dtype}]'


@export
class Stream:
    pending_tasks: ty.List[Task]
    stored_data: ty.Dict[str, StoredData]  # {dtypename: StoredData}
    final_target: str
    task_generators: ty.List[Job]
    this_process: psutil.Process
    threshold_mb = 1000

    def __init__(self, task_generators: ty.List[Job],
                 yield_outputs=None, max_workers=5):
        if isinstance(yield_outputs, str):
            yield_outputs = [yield_outputs]

        self.max_workers = max_workers
        self.task_generators = task_generators

        def get_priority(tg: Job):
            return tg.priority, tg.depth
        self.task_generators.sort(key=get_priority)

        self.pending_tasks = []
        self.this_process = psutil.Process(os.getpid())

        self.last_yielded_chunk = -1

        self.executors = dict(
            process=cf.ProcessPoolExecutor(max_workers=self.max_workers),
            thread=cf.ThreadPoolExecutor(max_workers=self.max_workers))
        # TODO: how to pass to taskgens?
        for tg in task_generators:
            tg.executors = self.executors

        who_wants = defaultdict(list)
        for tg in self.task_generators:
            if not len(tg.depends_on) and not tg.is_source:
                raise RuntimeError(
                    f"{tg} has no dependencies but it is not a source?")
            for dt in tg.depends_on:
                who_wants[dt].append(tg)
        for dt in yield_outputs:
            who_wants.setdefault(dt, [])

        self.stored_data = {
            dt: StoredData(dt, tgs, yield_to_user=dt in yield_outputs)
            for dt, tgs in who_wants.items()}

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
        if any([not tg.all_results_arrived
                for tg in self.task_generators]):
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
                result = task.generator.get_result(task)
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

        for tg in self.task_generators:
            if not tg.could_submit_new_task():
                continue

            # Are any of the required inputs exhausted?
            exhausted = False
            if tg.is_source:
                exhausted = tg.external_inputs_exhausted()
            else:
                for dt in tg.wants_input:
                    if dt in exhausted_dtypes:
                        exhausted = True

            # If so, start a cleanup task
            if exhausted:
                if tg.final_task_submitted:
                    continue
                inputs = {dt: self.stored_data[dt].slurp_for(tg)
                          for dt in tg.depends_on}
                return Signals.GOT_TASK, tg.get_cleanup_task(inputs)

            # Check external conditions satisfied
            if (not tg.external_input_ready()
                    and not tg.external_inputs_exhausted()):
                external_waits.append(tg)
                continue

            if tg.is_source:
                # Handle these separately (at the end) regardless of priority
                sources.append(tg)
                continue

            # Any input missing?
            missing_input = False
            for dtype, chunk_id in tg.wants_input.items():
                if not self.stored_data[dtype].has_stored(chunk_id):
                    requests_for[dtype] += 1
                    missing_input = True
                    break
            if missing_input:
                continue

            # We're good! Grab all inputs and submit.
            task_inputs = dict()
            for dtype, chunk_i in tg.wants_input.items():
                task_inputs[dtype] = self.stored_data[dtype].grab_for(
                    tg=tg, chunk_i=chunk_i)

            return Signals.GOT_TASK, tg.get_task(task_inputs)

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
        for tg in self.task_generators:
            print(f"{tg}:"
                  f"\n\twants {tg.wants_input}, "
                  f"\n\tlast_submitted {tg.last_submitted_i}, "
                  f"\n\thighest_cont._done {tg.highest_continuous_done_i}, "
                  f"\n\tfinal_task_submitted {tg.final_task_submitted},"
                  f"\n\tall_results_arrived {tg.all_results_arrived},"
                  f"\n\tis source {tg.is_source}")
        for dt, d in self.stored_data.items():
            d.print_status()
        for tg in self.task_generators:
            if not tg.final_task_submitted:
                try:
                    tg.cleanup(exception=exception, chunk_i=-1)
                except Exception as e:
                    print(f"Exceptional shutdown of {tg} failed")
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
        for tg in self.task_generators:
            if not tg.provides or not tg.all_results_arrived:
                continue
            for p in tg.provides:
                if not self.stored_data[p].n_stored():
                    exhausted.append(p)
        return exhausted
