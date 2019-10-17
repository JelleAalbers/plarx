import typing as ty
import time

import numpy as np

import plarx


class BasicSource(plarx.TaskGenerator):
    is_source = True
    provides = ('widgets',)

    n_chunks = 10

    def external_input_ready(self):
        return self.last_done_i < self.n_chunks - 1

    def external_inputs_exhausted(self):
        return self.last_done_i == self.n_chunks - 1

    def task_function(self, chunk_i, is_final=False, **kwargs):
        print(f"Source task {chunk_i} running. self.chunk_i is {self.chunk_i}")
        return dict(widgets=np.ones(42, dtype=np.float32) * chunk_i)

def test_source():
    """Single source, no processing"""
    sched = plarx.Scheduler(
        [BasicSource()],
        yield_output='widgets')
    i = 0
    for i, x in enumerate(sched.main_loop()):
        print(f"USER JUST GOT CHUNK {i}")
        assert i < 10
        np.testing.assert_array_equal(x, np.ones(42, dtype=np.float32) * i)
    assert i == 9


class BasicProc(plarx.TaskGenerator):
    provides = ('doodads',)
    depends_on = ('widgets',)

    def _task_function(self, chunk_i, is_final=False, **kwargs):
        return dict(doodads=np.ones(21, dtype=np.int) * chunk_i)


def test_process():
    """Source + processing"""
    sched = plarx.Scheduler(
        [BasicSource(), BasicProc()],
        yield_output='doodads')
    i = 0
    for i, x in enumerate(sched.main_loop()):
        print(f"USER JUST GOT CHUNK {i}")
        assert i < 10
        np.testing.assert_array_equal(x, np.ones(21, dtype=np.float32) * i)
    assert i == 9


class SecondSource(BasicSource):
    n_chunks = 3
    provides = ('thingies',)

    def task_function(self, chunk_i, is_final=False, **kwargs):
        print(f"YIELDING THINGIES # {chunk_i}")
        return dict(thingies=np.ones(42, dtype=np.float32) * chunk_i)


class FunnyCombination(plarx.TaskGenerator):
    n_chunks = 3
    depends_on = ('doodads', 'thingies')
    provides = ('gizmos',)
    changing_inputs = True

    toggle = False  # Will be switched to true on first call

    def task_function(self, chunk_i, is_final=False, **kwargs):
        bla = {k: kwargs[k][0] for k in kwargs}
        print(f"Gizmos task called wth {chunk_i}, {is_final}, {bla}")
        if is_final:
            return None
        self.toggle = not self.toggle
        print
        if self.toggle:
            # Request only widgets next time
            return (
                dict(gizmos=np.ones(11, dtype=np.int16) * chunk_i),
                ('doodads',))
        # Request widgets and doodads next time
        return (
            dict(gizmos=np.ones(11, dtype=np.int16) * chunk_i),
            ('doodads', 'thingies'))

    # 0: doodads 0 + thingies 0
    # 1: doodads 1
    # 2: doodads 2 + thingies 1
    # 3: doodads 3
    # 4: doodads 4 + thingies 2
    # 5: doodads 5
    # 6: doodads 6 + thingies 3
    # 7: doodads 7
    # NO 8: no more thingies available, rest of doodads discarded
    # TODO: NOT GOOD, want to at least see rest of doodads in final task


def test_complex():
    """Multiple dependencies, sources, and wants_input switching"""
    sched = plarx.Scheduler(
        [BasicSource(), BasicProc(), SecondSource(), FunnyCombination()],
        yield_output='gizmos')
    i = 0
    for i, x in enumerate(sched.main_loop()):
        print(f"USER JUST GOT CHUNK {i}")
        assert i < 8
        np.testing.assert_array_equal(x, np.ones(11, dtype=np.int16) * i)
    assert i == 7, sched.exit_with_exception(RuntimeError("OOPS"))
