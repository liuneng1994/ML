from collections import Counter
import numpy as np

import jieba


class Reader:
    def __init__(self, path):
        self._content = None
        self._path = path
        self._char_to_id = None
        self._id_to_char = None
        with open(path, encoding='utf-8') as f:
            self._content = ''.join(f.readlines())
        self.to_ids()

    @property
    def path(self):
        return self._path

    @property
    def content(self):
        return self._content

    @property
    def len(self):
        return len(self._char_to_id)

    def clear(self):
        del self._content

    def to_ids(self):
        body = self._content
        # print(Counter(body).items())
        if self._char_to_id is None:
            self._char_to_id = {key: no for no, (key, id) in enumerate(Counter(body).most_common())}
            self._id_to_char = list(self._char_to_id.keys())
        return self._char_to_id, self._id_to_char

    def gen_epochs(self, n, batch_size, num_steps):
        self.to_ids()
        raw_data = [self._char_to_id[char] for char in self._content]
        return self.ptb_iterator(list(raw_data)*n, batch_size, num_steps)

    def ptb_iterator(self, raw_data, batch_size, num_steps, steps_ahead=1):
        """Iterate on the raw PTB data.
        This generates batch_size pointers into the raw PTB data, and allows
        minibatch iteration along these pointers.
        Args:
          raw_data: one of the raw data outputs from ptb_raw_data.
          batch_size: int, the batch size.
          num_steps: int, the number of unrolls.
        Yields:
          Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
          The second element of the tuple is the same data time-shifted to the
          right by one.
        Raises:
          ValueError: if batch_size or num_steps are too high.
        """
        raw_data = np.array(raw_data, dtype=np.int32)

        data_len = len(raw_data)

        batch_len = data_len // batch_size

        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        offset = 0
        if data_len % batch_size:
            offset = np.random.randint(0, data_len % batch_size)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

        epoch_size = (batch_len - steps_ahead) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i * num_steps:(i + 1) * num_steps]
            y = data[:, i * num_steps + 1:(i + 1) * num_steps + steps_ahead]
            yield (x, y)

        if epoch_size * num_steps < batch_len - steps_ahead:
            yield (data[:, epoch_size * num_steps: batch_len - steps_ahead], data[:, epoch_size * num_steps + 1:])


if __name__ == '__main__':
    reader = Reader("三国演义.txt")
    assert reader.path == "三国演义.txt"
    # print(len(reader.content))
    print(len(reader.to_ids()))
