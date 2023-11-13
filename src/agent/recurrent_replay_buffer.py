from collections import namedtuple
import numpy as np
import torch

from util.functions import get_device


RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(get_device())


class RecurrentReplayBuffer:

    def __init__(
        self,
        o_dim,
        a_dim,
        sliding_win_len,
        capacity,
        batch_size,
        filler=0
    ):
        self.filler = filler

        # placeholders
        self.o = np.full((capacity, sliding_win_len + 1, o_dim), self.filler, dtype=np.float64)
        self.a = np.zeros((capacity, sliding_win_len, a_dim))
        self.r = np.zeros((capacity, sliding_win_len, 1))
        self.d = np.zeros((capacity, sliding_win_len, 1))
        self.m = np.zeros((capacity, sliding_win_len, 1))
        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers
        self.window_ptr = 0
        self.win_size_ptr = 0
        self.win_step_ptr = 0

        # trackers
        self.num_windows = 0

        # hyper-parameters
        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.batch_size = batch_size

        self.sliding_win_len = sliding_win_len


    def push(self, o, a, r, no, d):

        # zero-out current slot at the beginning of an episode
        tmp_ptr = (self.window_ptr + self.win_size_ptr) % self.capacity
        self.o[tmp_ptr] = self.filler
        self.a[tmp_ptr] = 0
        self.r[tmp_ptr] = 0
        self.d[tmp_ptr] = 0
        self.m[tmp_ptr] = 0
        self.ep_len[tmp_ptr] = 0
        self.ready_for_sampling[tmp_ptr] = 0


        # fill placeholders
        for i in range(self.win_size_ptr + 1):
            tmp_ptr = (self.window_ptr + i) % self.capacity
            self.o[tmp_ptr, self.win_size_ptr - i] = o
            self.a[tmp_ptr, self.win_size_ptr - i] = a
            self.r[tmp_ptr, self.win_size_ptr - i] = r
            self.d[tmp_ptr, self.win_size_ptr - i] = d
            self.m[tmp_ptr, self.win_size_ptr - i] = 1
            self.ep_len[tmp_ptr] += 1

        if d:

            # fill placeholders
            for i in range(self.win_size_ptr + 1):
                tmp_ptr = (self.window_ptr + i) % self.capacity
                self.o[tmp_ptr, self.win_size_ptr - i + 1] = no
                self.ready_for_sampling[tmp_ptr] = 1

            # update trackers
            if self.num_windows + self.win_size_ptr < self.capacity:
                self.num_windows += self.win_size_ptr
            elif self.num_windows < self.capacity:
                self.num_windows = self.capacity
                
            # reset pointers
            self.window_ptr = (self.window_ptr + self.win_size_ptr + 1) % self.capacity
            self.win_size_ptr = 0

        else:

            # update pointers
            if self.win_size_ptr + 1 < self.sliding_win_len:
                self.win_size_ptr += 1
            else:
                self.o[self.window_ptr, self.win_size_ptr + 1] = no
                self.ready_for_sampling[self.window_ptr] = 1
                if self.num_windows < self.capacity:
                    self.num_windows += 1
                self.window_ptr = (self.window_ptr + 1) % self.capacity


    def sample(self):

        assert self.num_windows >= self.batch_size, f"Number of windows: {self.num_windows} >= Batch size: {self.batch_size}"

        # sample episode indices
        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=self.batch_size)

        # convert choices to tensors on the right device
        o = as_tensor_on_device(self.o[choices]).view(self.batch_size, self.sliding_win_len + 1, self.o_dim)
        a = as_tensor_on_device(self.a[choices]).view(self.batch_size, self.sliding_win_len, self.a_dim)
        r = as_tensor_on_device(self.r[choices]).view(self.batch_size, self.sliding_win_len, 1)
        d = as_tensor_on_device(self.d[choices]).view(self.batch_size, self.sliding_win_len, 1)
        m = as_tensor_on_device(self.m[choices]).view(self.batch_size, self.sliding_win_len, 1)

        return RecurrentBatch(o, a, r, d, m)

