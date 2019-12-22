import concurrent.futures
from functools import partial
import typing as ty
from types import MethodType

import plarx

from .common import exporter
export, __all__ = exporter()


class CannotSubmitNewTask(Exception):
    pass


@export
class Task(ty.NamedTuple):
    """Task object, tracking a future submitted to a pool"""

    # If true, mark Job as finished on completion
    is_final: bool

    # Generator that made the task.
    job: ty.Any       # Wait for python 4.0 / future annotations

    # Chunk number about to be produced
    chunk_i: int

    # Future object. Result will be {dtypename: array, ...}
    # or possibly, and only for the final task, None.
    #
    # For tasks that are placeholders for data to be returned to the user,
    # future directly stores the result.
    future: concurrent.futures.Future

    def __repr__(self):
        return f"Task[{self.job}:{self.chunk_i}]"


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
        self = type('Job' + plarx.random_str(10),
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
        return f"Job[{_from}>{_to}]"

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

    def _submit_task(self, content, task_i, is_final) -> Task:
        future = self.executors[self.submit_to].submit(content)
        self.last_submitted_i += 1
        return Task(
            future=future,
            job=self,
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
            -> ty.Dict[str, ty.Any]:
        raise NotImplementedError

    def cleanup(self, chunk_i: int, exception=None, **inputs) \
            -> ty.Union[None, ty.Dict[str, ty.Any]]:
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
