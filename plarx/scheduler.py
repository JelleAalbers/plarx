from collections import defaultdict
import concurrent.futures as cf
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
        self.seen_input = self.wants_input.copy()

    def __repr__(self):
        _from = _to = ''
        if len(self.wants_input):
            _from = list(self.wants_input.keys())[0]
        if len(self.provides):
            _to = self.provides[0]
        return f"[{_from}>{_to}]"

    def get_task(self, inputs=None, is_final=False) -> Task:
        """Submit a task to operate on inputs"""
        if not self.parallel and self.chunk_i != self.last_done_i:
            raise RuntimeError(f"Attempt to get task for {self} "
                               f"out of order: {self.chunk_i}, last done is "
                               f"{self.last_done_i}.")
        assert not self.all_results_arrived, f"Can't get {self} task: all results already arrived"
        assert not self.final_task_submitted, f"Can't get {self} task: final task already submitted"

        self.chunk_i += 1

        if is_final or self.is_source:
            assert inputs is None, "Passed inputs to source or final task"
            if self.submit_to == 'user':
                content = None
            else:
                content = partial(self._task_function, chunk_i=self.chunk_i)
        else:
            # Validate inputs
            for k in inputs:
                if not isinstance(inputs[k], np.ndarray):
                    raise RuntimeError(f"Got {type(inputs[k])} instead of np"
                                       f"array as input {k} given to {self}")
                if k not in self.wants_input:
                    raise RuntimeError(f"Unwanted input {k} given to {self}")
                self.seen_input[k] += 1
            for k in self.wants_input:
                if k not in inputs:
                    raise RuntimeError(f"Missing input {k} to {self}")

            if self.submit_to == 'user':
                assert len(self.wants_input) == 1
                content = list(inputs.values())[0]
            else:
                content = partial(self._task_function,
                                  chunk_i=self.chunk_i,
                                  **inputs)
            if not self.parallel:
                # Request the next set of inputs
                for dt, i in self.wants_input.items():
                    self.wants_input[dt] = i + 1

        if is_final:
            self.final_task_submitted = True

        return Task(content=content,
                    generator=self,
                    chunk_i=self.chunk_i,
                    is_final=is_final)

    def get_result(self, task: Task):
        # Update accounting
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
            self.wants_input = {dt: self.seen_input[dt] for dt in new_inputs}

        # Check and return result
        if result is None:
            assert task.is_final, f"{task} is not final but returned None"
        else:
            if self.submit_to == 'user':
                assert isinstance(result, np.ndarray),\
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
        return result

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
        self.processpool = cf.ProcessPoolExecutor(max_workers=self.max_workers)
        self.threadpool = cf.ThreadPoolExecutor(max_workers=self.max_workers)

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
        print(self.task_generators, self.stored_data)

    def main_loop(self):
        while True:
            self._receive_from_done_tasks()
            task = self._get_new_task()
            if task is None:
                # No more work, except pending tasks
                # and tasks that may follow from their results.
                if not self.pending_tasks:
                    if all([tg.all_results_arrived
                            for tg in self.task_generators]):
                        break  # All done. We win!
                    self.exit_with_exception(RuntimeError(
                        "No available or pending tasks, "
                        "but data is not exhausted!"))
                # Wait for a pending task to complete
            else:
                # print(f"Submitting task {task}")
                if task.submit_to == 'user':
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
        self.threadpool.shutdown()
        self.processpool.shutdown()

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
        if task.submit_to == 'thread':
            f = self.threadpool.submit(task.content)
        elif task.submit_to == 'process':
            f = self.processpool.submit(task.content)
        else:
            raise RuntimeError(f"Invalid submission target {task.submit_to}")
        task.future = f
        self.pending_tasks += [task]

    def _get_new_task(self):
        external_waits = []    # TaskGenerators waiting for external conditions
        sources = []           # Sources we could load more data from
        requests_for = defaultdict(int)  # Requests for particular inputs

        # Which inputs are exhausted?
        # Note final tasks can return results, so we check for
        # all_results_arrived, not final_task_submitted!
        exhausted_inputs = sum([list(tg.provides)
                                for tg in self.task_generators
                                if tg.all_results_arrived],
                               [])

        for tg in self.task_generators:
            if tg.final_task_submitted:
                continue

            # If not parallelizable, check previous task has completed
            if not tg.parallel and tg.last_done_i < tg.chunk_i:
                # print(f"{tg} waiting on completion of last task. "
                #       f"Last done: {tg.last_done_i}, chunk_i {tg.chunk_i}")
                continue        # Need previous task to finish first

            # Are the inputs exhausted?
            if ((not tg.is_source or tg.external_inputs_exhausted())
                    and all([dt in exhausted_inputs for dt in tg.depends_on])):
                # print(f"inputs exhausted for {tg}")
                if not tg.final_task_submitted:
                    return tg.get_task(is_final=True)
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

            # Are the required inputs available?
            inputs_complete = True
            for dtype, chunk_id in tg.wants_input.items():
                if chunk_id not in self.stored_data[dtype].stored:
                    # Need input to arrive first
                    inputs_complete = False
                    requests_for[dtype] += 1
                    break
            if not inputs_complete:
                continue

            # Yes! Submit the task
            task_inputs = dict()
            for dtype, chunk_i in tg.wants_input.items():
                self.stored_data[dtype].seen_by_consumers[tg] = chunk_i
                task_inputs[dtype] = self.stored_data[dtype].stored[chunk_i]
            self._cleanup_cache()

            task = tg.get_task(task_inputs)
            return task

        if sources:
            # No computation tasks to do, but we could load new data
            if (self.this_process.memory_info().rss / 1e6 > self.threshold_mb
                    and self.pending_tasks):
                # ... Let's not though; instead wait for current tasks.
                # (We could perhaps also wait for an external condition
                # but in all likelihood a task will complete soon enough)
                return None
            # Load data for the source that is blocking the most tasks
            # Jitter it a bit for better performance on ties..
            # TODO: There is a better way, but I'm too lazy now
            requests_for_source = [(s, sum([requests_for.get(dt, 0)
                                            for dt in s.provides]) + random())
                                   for s in sources]
            s, _ = max(requests_for_source, key=lambda q: q[1])
            return s.get_task(None)

        if external_waits:
            # We could wait for an external condition...
            if len(self.pending_tasks):
                # ... but probably an existing task will complete first
                return None
            # ... so let's do that.
            self._emit_status(f"{external_waits} waiting on external condition")
            time.sleep(5)
            # TODO: for very long waits this will trip the recursion limit!
            return self._get_new_task()

        # No work to do. Maybe a pending task will still generate some though.
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
