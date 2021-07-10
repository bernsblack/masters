class WalkForwardIndexer:
    def __init__(self, total_set_size, test_set_size, sub_set_size):
        """
        Jump each step by test_set size so that each value of the dataset can be tested
        :param total_set_size: total length of all data points
        :param test_set_size: test set size per step, also the step size
        :param sub_set_size: size of a subset of the data set, includes train, val and test set sizes
        """
        self.total_length = total_set_size
        self.step_size = test_set_size  # sub_set_size - test_set_size
        self.frame_size = sub_set_size

        self.start_index = 0
        self.stop_index = self.frame_size

    def __iter__(self):
        self.start_index = 0
        self.stop_index = self.frame_size
        return self

    def __next__(self):
        if self.stop_index > self.total_length:
            raise StopIteration
        else:
            temp_start_index = self.start_index
            temp_stop_index = self.stop_index

            self.start_index += self.step_size
            self.stop_index += self.step_size

            return temp_start_index, temp_stop_index
