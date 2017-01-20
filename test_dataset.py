import numpy as np
from utils.dataset import DatasetFeed


def dummy_dataset(num_data_points, num_subblocks, subblock_sizes):
    """
    Creates a dummy dataset using randomly drawn N(0, 1) samples
    :param num_data_points: Number of points in the dataset
    :param num_subblocks: Number of sub-data-blocks for each data point (e.g. xtfreq, xtmag etc)
    :param subblock_sizes: list of shapes for subblocks. These should be np.hstack-able
    :return:
    """
    assert len(subblock_sizes)==num_subblocks
    data = []
    for data_point_index in range(num_data_points):
        blocks = []
        for block_index in range(num_subblocks):
            blocks.append(np.random.randn(*subblock_sizes[block_index]))
        data.append(blocks)
    return data

def dummy_dataset_with_different_shapes(num_data_points_list, num_subblocks, subblock_sizes_list):
    data = []
    for i in range(len(num_data_points_list)):
        datatemp = dummy_dataset(num_data_points_list[i], num_subblocks, subblock_sizes_list[i])
        data.extend(datatemp)
    return data

data = dummy_dataset_with_different_shapes([3, 2], 2, [[(2,2), (2,2)], [(1, 1), (1, 1)]])
print("Dummy data:")
for datapoint in data:
    print(datapoint)

datasetfeed = DatasetFeed(data, 2)
print('Dummy data after concatenating arrays:')
for datapoint in datasetfeed.data:
    print(datapoint)
datasetfeed.set_all_data_blocks_to_max_shape()
print("Dummy data after padding and creating masks:")
for i, datapoint in enumerate(datasetfeed.data):
    print(datapoint)
    print(datasetfeed.data_masks[i])