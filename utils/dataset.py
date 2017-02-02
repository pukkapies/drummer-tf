import numpy as np
from random import shuffle


class DatasetFeed(object):

    def __init__(self, loaded_list, minibatch_size):
        """
        Initialiser
        :param loaded_list: List of lists, which contains numpy arrays for training
        """
        assert type(loaded_list) == list
        self.data = self.concatenate_npy_arrays(loaded_list)
        self.num_data_points = len(self.data)
        print("Number of data points loaded: ", self.num_data_points)
        self.max_data_shape, self.ndim = self.get_max_data_shape()
        print("max_data_shape: {}, ndim: {}".format(self.max_data_shape, self.ndim))
        self.minibatch_size = minibatch_size
        assert self.minibatch_size <= len(self.data), "Data minibatch must be less than the number of data points"
        self.epochs_completed = 0
        self.current_dataset_index = 0

    def concatenate_npy_arrays(self, loaded_list):
        """Concatenates each sublist corresponding to a single audio file into one numpy array block"""
        data = []
        for i in range(len(loaded_list)):
            data_block = np.hstack(loaded_list[i])
            data.append(data_block)
        return data

    def get_max_data_shape(self, axes=None, assert_all_shapes_are_same=False):
        """
        Gets the max shape (independently for each dimension) of each data element.
        Asserts the number of dimensions are the same for each data element.
        :param axes: List of axes to normalise. If None, all axes are normalised.
        :param assert_all_shapes_are_same: Whether to assert that all shapes are the same
        :return: Tuple: Each element of the tuple is the maximum size over all data elements
                        Axes that are not to be normalised have zero values
        """
        assert type(self.data) == list
        # Get shape and rank of first element
        ndim = self.data[0].ndim
        if axes:
            assert len(axes) == ndim
        else:
            axes = list(range(ndim))  # All axes
        shape = self.data[0].shape
        max_dim_size = [0] * ndim
        for data_object in self.data:
            # print(data_object.shape)
            assert data_object.ndim == ndim, "Data objects do not all have the same rank"
            if assert_all_shapes_are_same:
                assert data_object.shape == shape, "Data objects are not all the same shape"
            for dim in range(ndim):
                if dim not in axes:
                    continue
                if data_object.shape[dim] > max_dim_size[dim]:
                    max_dim_size[dim] = data_object.shape[dim]
        return tuple(max_dim_size), ndim

    def set_all_data_blocks_to_max_shape(self, pad_with=0.):
        """
        Makes all data blocks the same shape by padding with pad_with. Stores a list of masks (same length as
        self.data), where each mask is zero for all entries corresponding to padded values and
        one otherwise
        :param pad_with: Value to pad the data blocks
        :return: None
        """
        assert type(self.data) == list
        masks = []
        for i, data_element in enumerate(self.data):
            # First create the mask
            mask = np.zeros(self.max_data_shape)
            slices = [slice(0, data_element.shape[j]) for j in range(self.ndim)]
            mask[slices] = 1.
            masks.append(mask)

            pad_list = [(0, self.max_data_shape[i] - data_element.shape[i]) for i in range(self.ndim)]
            # Pads the end of each dimension with zeros:
            self.data[i] = np.pad(data_element, pad_list, 'constant', constant_values=(pad_with, pad_with))
        self.data_masks = masks

    def next_batch(self, shuffle_after_every_epoch=True):
        """
        Returns the next minibatch
        :return: np.ndarray, shape (batch_size, data_shape)
        """
        current_index = self.current_dataset_index
        next_index = self.current_dataset_index + self.minibatch_size
        if next_index < self.num_data_points:  # next_index still points within the range of data points
            self.current_dataset_index = next_index
            return np.asarray(self.data[current_index: next_index])
        else:
            self.current_dataset_index = next_index % self.num_data_points
            self.epochs_completed += 1
            print("Completed {} epochs".format(self.epochs_completed))
            first_sub_batch = self.data[current_index:]  # The remainder of the current set of data points
            if shuffle_after_every_epoch:
                shuffle(self.data)
            return np.asarray(first_sub_batch + self.data[:self.current_dataset_index])


class DatasetPreloaded(object):
    #TODO: class for preloading a dataset as a variable, so it is stored on the GPU
    #TODO: see (https://www.tensorflow.org/versions/r0.11/how_tos/reading_data/#preloaded_data)
    pass


