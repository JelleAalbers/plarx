from collections import defaultdict
import concurrent.futures as cf
from enum import Enum
from functools import partial
import os
from random import random
import time
import typing as ty

import psutil
import numpy as np

from .utils import exporter
export, __all__ = exporter()


class Task:
    """Task object, tracking a future submitted to a pool

    :param content: Task to submit to thread/process pool
    or data to yield to user
    :param is_final: If true, mark TaskGenerator as finished on completion
    :param generator: Generator that made the task (need python 3.7
    to annotate properly)
    :param future: Future object. Result will be {dtypename: array, ...}
    or possibly, and only for the final task, None. Usually set later.
    :param chunk_i: Chunk number about to be produced
    """
    content: ty.Union[np.ndarray, ty.Callable]
    is_final: False
    generator: ty.Any
    provides: ty.Tuple[str]
    submit_to: str
    future: cf.Future = None
    chunk_i = 0

    def __init__(self, content, generator, chunk_i, is_final=False,
                 future=None):
        self.chunk_i = chunk_i
        self.generator = generator
        self.provides = self.generator.provides
        self.submit_to = self.generator.submit_to
        self.content = content
        self.is_final = is_final
        self.future = future

    def __repr__(self):
        return f"{self.generator}:{self.chunk_i}"


class CannotSubmitNewTask(Exception):
    pass


@export
class TaskGenerator:
    provides: ty.Tuple[str] = tuple()     # Produced data types
    depends_on: ty.Tuple[str] = tuple()   # Input data types
    wants_input: ty.Dict[str, int]  # [(dtype, chunk_i), ...] of inputs
                                    # needed to make progress
    seen_input: ty.Dict[str, int]   # [(dtype, chunk_i), ...] of inputs
                                    # already seen
    is_source = False           # True if loader/producer of data without deps
    submit_to = 'thread'        # thread, process, or user
    parallel = False            # Can we start more than one task at a time?
    changing_inputs = False     # If True, task_function returns
                                # (result, inputs_wanted_next_time)
                                # Cannot parallelize.

    priority = 0                # 0 = saver/target, 1 = source, 2 = other
    depth = 0                   # Dependency hops to final target

    chunk_i = -1                # Chunk number of last emitted task
    last_done_i = -1            # Chunk number of last completed task
    input_cache: ty.Dict[str, np.ndarray]
                                # Inputs we could not yet pass to computation

    # "Finished" variables. Note there are two...
    final_task_submitted = False
    all_results_arrived = False

    def __init__(self):
        if self.changing_inputs and self.parallel:
            raise RuntimeError("Cannot parallelize a TaskGenerator with "
                               "indirect input delivery.")
        if self.is_source and len(self.depends_on):
            raise RuntimeError("A source cannot depend on any data type")
        self.wants_input = {dt: 0 for dt in self.depends_on}
        self.seen_input = {dt: -1 for dt in self.depends_on}

    def __repr__(self):
        _from = _to = ''
        if len(self.wants_input):
            _from = list(self.wants_input.keys())[0]
        if len(self.provides):
            _to = self.provides[0]
        return f"[{_from}>{_to}]"

    def get_task(self, inputs=None, is_final=False) -> Task:
        """Submit a task to operate on inputs"""
        self._check_can_submit()
        self.chunk_i += 1

        if is_final or self.is_source:
            assert inputs is None, "Passed inputs to source or final task"
            if self.submit_to == 'user':
                content = None
            else:
                content = partial(self._task_function, chunk_i=self.chunk_i)
        else:
            self._validate_inputs(inputs)
            for k in inputs:
                self.seen_input[k] += 1

            if self.submit_to == 'user':
                assert len(self.wants_input) == 1
                content = list(inputs.values())[0]
            else:
                content = partial(self._task_function,
                                  chunk_i=self.chunk_i,
                                  **inputs)

            if not self.changing_inputs:
                # Request the next set of inputs
                for dt in self.wants_input:
                    self.wants_input[dt] += 1

        if is_final:
            self.final_task_submitted = True

        if self.submit_to in self.executors:
            future = self.executors[self.submit_to].submit(content)
        elif self.submit_to == 'user':
            future = None
        else:
            raise RuntimeError(f"Invalid submission target {task.submit_to}")

        return Task(content=content,
                    future=future,
                    generator=self,
                    chunk_i=self.chunk_i,
                    is_final=is_final)

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
        if not self.parallel and self.chunk_i != self.last_done_i:
            raise CannotSubmitNewTask(
                f"Attempt to get task for {self} "
                f"out of order: {self.chunk_i}, last done is {self.last_done_i}.")
        if self.all_results_arrived:
            raise CannotSubmitNewTask(
                f"Can't get {self} task: all results already arrived")
        if self.final_task_submitted:
            raise CannotSubmitNewTask(
                "Can't get {self} task: final task already submitted")

    def get_result(self, task: Task):
        """Return result of task"""
        assert self is task.generator, "Hey, that's not my task!"
        if task.is_final:
            # Note we set this BEFORE fetching raising exception on a failed
            # result, so we do not retry a failed finishing task.
            self.all_results_arrived = True
        else:
            self.last_done_i = task.chunk_i

        # Fetch result
        if self.submit_to == 'user':
            result = task.content
        else:
            if not task.future.done():
                raise RuntimeError("get_result called before task was done")
            result = task.future.result()  # Will raise if exception

        # Record new inputs
        if self.changing_inputs:
            assert isinstance(result, tuple) and len(result) == 2, \
                f"{self} changes inputs but didn't return a two-tuple"
            result, new_inputs = result
            self.wants_input = {dt: self.seen_input[dt] + 1
                                for dt in new_inputs}

        # Check and return result
        if result is None:
            assert task.is_final, f"{task} is not final but returned None"
        else:
            self._validate_results(task, result)
        return result

    def _validate_inputs(self, inputs: ty.Dict[str, np.ndarray]):
        """Check if correct inputs dict is provided"""
        for k in inputs:
            if not isinstance(inputs[k], np.ndarray):
                raise RuntimeError(f"Got {type(inputs[k])} instead of np"
                                   f"array as input {k} given to {self}")
            if k not in self.wants_input:
                raise RuntimeError(f"Unwanted input {k} given to {self}")
        for k in self.wants_input:
            if k not in inputs:
                raise RuntimeError(f"Missing input {k} to {self}")

    def _validate_results(self, task, result):
        """Check if task returned the correct result"""
        if self.submit_to == 'user':
            assert isinstance(result, np.ndarray), \
                f"Attempt to yield a {type(result)} rather than a " \
                f"numpy array to the user"
        else:
            assert isinstance(result, dict), \
                f"{task} returned a {type(result)} rather than a dict"
            for k in result:
                assert k in self.provides, \
                    f"{task} provided unwanted output {k}"
            for k in self.provides:
                assert k in result, \
                    f"{task} failed to provide needed output {k}"

    def _task_function(self, chunk_i: int, is_final=False, **kwargs)\
            -> ty.Dict[str, np.ndarray]:
        result = self.task_function(
            chunk_i=chunk_i, is_final=is_final, **kwargs)
        if is_final:
            self.cleanup()
        return result

    # Function to override

    def task_function(self, chunk_i: int, is_final=False, **kwargs)\
            -> ty.Dict[str, np.ndarray]:
        raise NotImplementedError

    def external_inputs_exhausted(self):
        """For sources, return whether external inputs exhausted"""
        return False

    def external_input_ready(self):
        """For sources, return whether the next external input
         (i.e. self.chunk_id + 1) is ready"""
        return True

    def cleanup(self, exception=None):
        """Execute final cleanup, e.g. closing of files.
        Does not return result -- that's for the task function
        with is_final = True.
        """
        pass


class StoredData:
    dtype: str                  # Name of data type
    stored: ty.Dict[int, np.ndarray]
                                # [(chunk_i, data), ...]
    seen_by_consumers: ty.Dict[TaskGenerator, int]
                                # Last chunk seen by each of the generators
                                # that need it
    last_contiguous = -1        # Latest chunk that has arrived
                                # for which all previous chunks have arrived

    def __init__(self, dtype, wanted_by: ty.List[TaskGenerator]):
        self.dtype = dtype
        self.stored = dict()
        self.seen_by_consumers = {tg: -1
                                  for tg in wanted_by}


class GetTaskStatus(Enum):
    NO_TASK = 0
    GOT_TASK = 1
    YIELD_TO_USER = 2
    WAIT_EXTERNAL = 3


@export
class Scheduler:
    pending_tasks: ty.List[Task]
    stored_data: ty.Dict[str, StoredData]  # {dtypename: StoredData}
    final_target: str
    task_generators: ty.List[TaskGenerator]
    this_process: psutil.Process
    threshold_mb = 1000

    def __init__(self, task_generators: ty.List[TaskGenerator],
                 yield_output=None, max_workers=5):
        self.max_workers = max_workers
        if yield_output:
            yielder = type('YieldOutputTaskGenerator',
                           (TaskGenerator,),
                           dict(depends_on=(yield_output,),
                                submit_to='user'))()
            task_generators = task_generators + [yielder]
        self.task_generators = task_generators
        self.task_generators.sort(key=lambda _tg: (_tg.priority, _tg.depth))
        self.pending_tasks = []
        self.this_process = psutil.Process(os.getpid())

        self.executors = dict(
            process=cf.ProcessPoolExecutor(max_workers=self.max_workers),
            thread=cf.ThreadPoolExecutor(max_workers=self.max_workers))
        # TODO: how to pass to taskgens?
        for tg in task_generators:
            tg.executors = self.executors

        self.who_wants = defaultdict(list)
        for tg in self.task_generators:
            if not len(tg.depends_on) and not tg.is_source:
                raise RuntimeError(
                    f"{tg} has no dependencies but it is not a source?")
            for dt in tg.depends_on:
                self.who_wants[dt].append(tg)
        self.stored_data = {
            dt: StoredData(dt, tgs)
            for dt, tgs in self.who_wants.items()}

    def main_loop(self):
        while True:
            self._receive_from_done_tasks()
            status, task = self._get_new_task()

            if status is GetTaskStatus.WAIT_EXTERNAL:
                self._emit_status(f"{external_waits} waiting on external condition")
                time.sleep(5)
                continue   # Try finding a task again

            if task is not None and task.generator.submit_to == 'user':
                status = GetTaskStatus.YIELD_TO_USER

            if status is GetTaskStatus.NO_TASK:
                # No work right now.
                if not self.pending_tasks:
                    if all([tg.all_results_arrived
                            for tg in self.task_generators]):
                        break  # All done. We win!
                    self.exit_with_exception(RuntimeError(
                        "No available or pending tasks, "
                        "but data is not exhausted!"))
                # Wait for a pending task to complete
            else:
                if status is GetTaskStatus.YIELD_TO_USER:
                    # This is not a real task, we just have to submit
                    # a piece of the final target to the user
                    result = self._get_task_result(task)
                    if result is None:
                        assert task.is_final
                    else:
                        yield result
                    continue            # Find another task
                else:
                    self._submit_task(task)
                if len(self.pending_tasks) < self.max_workers:
                    continue            # Find another task
            self.wait_until_task_done()
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

    def _emit_status(self, msg):
        print(msg)
        print(f"\tPending tasks: {self.pending_tasks}")

    def _receive_from_done_tasks(self):
        still_pending = []
        for task in self.pending_tasks:
            f = task.future
            if not f.done():
                still_pending.append(task)
                continue
            result = self._get_task_result(task)
            if result is None:
                continue
            for dtype, result in result.items():
                if dtype not in self.stored_data:
                    # TODO: consider: can be a bug but can also happen
                    print(f"Got {dtype} which nobody wants, discarding...")
                    continue
                d = self.stored_data[dtype]
                d.stored[task.chunk_i] = result
                if d.last_contiguous == task.chunk_i - 1:
                    d.last_contiguous += 1
        self.pending_tasks = still_pending

    def _get_task_result(self, task):
        try:
            return task.generator.get_result(task)
        except Exception as e:
            self.exit_with_exception(e, f"Exception from {task}")

    def _submit_task(self, task: Task):
        self.pending_tasks += [task]

    def _get_new_task(self):
        """Return a new task, or None, if it is wiser or necessary to wait"""
        external_waits = []    # TaskGenerators waiting for external conditions
        sources = []           # Sources we could load more data from
        requests_for = defaultdict(int)  # Requests for particular inputs
        exhausted_dtypes = self._get_exhausted()

        for tg in self.task_generators:
            if not tg.could_submit_new_task():
                continue

            # Are the inputs exhausted?
            if ((not tg.is_source or tg.external_inputs_exhausted())
                    and all([dt in exhausted_dtypes for dt in tg.depends_on])):
                if not tg.final_task_submitted:
                    return GetTaskStatus.GOT_TASK, tg.get_task(is_final=True)
                # Final task submitted, cannot do anything else
                continue

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
            missing_input = self._get_missing_input(tg)
            if missing_input is not None:
                requests_for[missing_input] += 1
                continue

            # We're good! Submit the task
            task_inputs = dict()
            for dtype, chunk_i in tg.wants_input.items():
                self.stored_data[dtype].seen_by_consumers[tg] = chunk_i
                task_inputs[dtype] = self.stored_data[dtype].stored[chunk_i]
            self._cleanup_cache()

            task = tg.get_task(task_inputs)
            return GetTaskStatus.GOT_TASK, task

        if sources:
            # No computation tasks to do, but we could load new data
            if (self.this_process.memory_info().rss / 1e6 > self.threshold_mb
                    and self.pending_tasks):
                # ... Let's not though; instead wait for current tasks.
                # (We could perhaps also wait for an external condition
                # but in all likelihood a task will complete soon enough)
                return GetTaskStatus.NO_TASK, None
            # Load data for the source that is blocking the most tasks
            # Jitter it a bit for better performance on ties..
            # TODO: There is a better way, but I'm too lazy now
            requests_for_source = [(s, sum([requests_for.get(dt, 0)
                                            for dt in s.provides]) + random())
                                   for s in sources]
            s, _ = max(requests_for_source, key=lambda q: q[1])
            return GetTaskStatus.GOT_TASK, s.get_task(None)

        if external_waits:
            if len(self.pending_tasks):
                # Assume an existing task will complete before the wait time.
                # TODO: Good idea? Maybe make configurable?
                return GetTaskStatus.NO_TASK, None
            else:
                return GetTaskStatus.WAIT_EXTERNAL, None

        # No work to do. Maybe a pending task will still generate some though.
        return GetTaskStatus.NO_TASK, None

    def _get_missing_input(self, tg):
        """Return a datatype for which the chunk needed by tg is not available,
        or None."""
        for dtype, chunk_id in tg.wants_input.items():
            if chunk_id not in self.stored_data[dtype].stored:
                return dtype
        return None

    def _cleanup_cache(self):
        """Remove any data from our stored_data that has been seen
        by all the consumers"""
        for d in self.stored_data.values():
            if not len(d.seen_by_consumers):
                raise RuntimeError(f"{d} is not consumed by anyone??")
            seen_by_all = min(d.seen_by_consumers.values())
            d.stored = {chunk_i: data
                        for chunk_i, data in d.stored.items()
                        if chunk_i > seen_by_all}

    def exit_with_exception(self, exception, extra_message=''):
        print(extra_message)
        for tg in self.task_generators:
            print(f"{tg}:"
                  f"\n\twants {tg.wants_input}, "
                  f"\n\tat chunk {tg.chunk_i}, "
                  f"\n\tfinished {tg.all_results_arrived},"
                  f" is source {tg.is_source}")
        for dt, d in self.stored_data.items():
            print(f"{dt}: stored: {list(d.stored.keys())}, "
                  f"last_contiguous: {d.last_contiguous}, "
                  f"seen: {d.seen_by_consumers}")
        for tg in self.task_generators:
            if not tg.all_results_arrived:
                try:
                    tg.cleanup(exception=exception)
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
        return sum([list(tg.provides)
                    for tg in self.task_generators
                    if tg.all_results_arrived],
                    [])
