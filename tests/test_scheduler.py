import typing as ty
import time

import numpy as np

import plarx


class BasicSource(plarx.TaskGenerator):
    is_source = True
    provides = ('widgets',)

    def external_input_ready(self):
        return self.last_done_i < 9

    def external_inputs_exhausted(self):
        return self.last_done_i == 9

    def task_function(self, chunk_i: int, is_final=False, **kwargs) \
            -> ty.Dict[str, np.ndarray]:
        print(f"Source task {chunk_i} running. self.chunk_i is {self.chunk_i}")
        return dict(widgets=np.ones(42, dtype=np.float32) * chunk_i)


def test_simple():
    sched = plarx.Scheduler([BasicSource()], yield_output='widgets')
    for i, x in enumerate(sched.main_loop()):
        print(f"USER JUST GOT CHUNK {i}")
        assert i < 10
        np.testing.assert_array_equal(x, np.ones(42, dtype=np.float32) * i)
    assert i == 9
