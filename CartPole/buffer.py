import random

from collections import deque


class Buffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    @property
    def size(self):
        return len(self.buffer)
