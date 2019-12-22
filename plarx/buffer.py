import typing as ty
import numpy as np

from .common import exporter
export, __all__ = exporter()

import plarx

@export
class Buffer:
    _dtype: str                 # Name of data type
    _stored: ty.Dict[int, np.ndarray]
                                # [(chunk_i, data), ...]
    _seen_by_consumers: ty.Dict[plarx.Job, int]
                                # Last chunk seen by each of the tasks
                                # that need it
    _last_contiguous = -1       # Latest chunk that has arrived
                                # for which all previous chunks have arrived

    _yielding_to_user = False     # Output must be yielded to the user
    _last_yielded_to_user = -1    # Last chunk yielded to the user

    def __init__(self,
                 dtype,
                 wanted_by: ty.List[plarx.Job],
                 yield_to_user=False, ):
        """Buffer of stored data of one type.

        :param dtype: Name of the data type to be stored
        :param wanted_by: List of jobs that want this data type
        :param yield_to_user: if True (default False), this data type
        will be yielded to the user.
        """
        self._dtype = dtype
        self._stored = dict()
        self._seen_by_consumers = {tg: -1
                                   for tg in wanted_by}
        self._yielding_to_user = yield_to_user

    def add(self, data: np.ndarray, chunk_i: int):
        """Add a new chunk of data to the buffer"""
        self._stored[chunk_i] = data
        if self._last_contiguous == chunk_i - 1:
            self._last_contiguous += 1

    def grab_for(self, chunk_i, job) -> np.ndarray:
        """Return chunk_i for use in job"""
        assert self._seen_by_consumers[job] < chunk_i, \
            f"Cannot get {chunk_i} for {job}, " \
            f"it has already seen {self._seen_by_consumers[job]}"
        assert chunk_i in self._stored, \
            f"Cannot get {chunk_i} for {job}, have only {self._stored.keys()}"

        self._seen_by_consumers[job] = chunk_i
        return self._stored[chunk_i]

    def slurp_for(self, job) -> ty.List[np.ndarray]:
        """Return list of all buffered chunks not yet seen by job"""
        result = []
        while self._seen_by_consumers[job] < self._last_contiguous:
            result.append(self.grab_for(
                job=job,
                chunk_i=self._seen_by_consumers[job] + 1))
        return result

    def yield_to_user(self):
        """Yield to user all buffered datatypes it has not yet seen"""
        if not self._yielding_to_user:
            return

        chunk_i = self._last_yielded_to_user + 1
        while chunk_i in self._stored:
            result = self._stored[chunk_i]
            if not isinstance(result, np.ndarray):
                raise ValueError(
                    f"Attempt to yield a {type(result)} rather "
                    f"than a numpy array to the user")
            yield result  #, self.dtype, chunk_i
            self._last_yielded_to_user = chunk_i
            chunk_i += 1

    def cleanup(self):
        """Remove chunks seen by all consumers (and user, if yielding to user)
        from the buffer
        """
        seen_by_all = min(self._seen_by_consumers.values(),
                          default=float('inf'))
        if self._yielding_to_user:
            seen_by_all = min(seen_by_all, self._last_yielded_to_user)
        elif not len(self._seen_by_consumers):
            raise RuntimeError(f"{self._dtype} is not consumed by anyone??")

        self._stored = {chunk_i: data
                        for chunk_i, data in self._stored.items()
                        if chunk_i > seen_by_all}

    def print_status(self):
        print(f"{self._dtype}: stored: {list(self._stored.keys())}, "
              f"last_contiguous: {self._last_contiguous}, "
              f"seen: {self._seen_by_consumers}")

    def has_stored(self, chunk_id):
        return chunk_id in self._stored

    def n_stored(self):
        return len(self._stored)

    def __repr__(self):
        return f'Buffer[{self._dtype}]'
