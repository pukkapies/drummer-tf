import tensorflow as tf
import numpy as np

def setup_training_data(loaded_data, batch_size):
    """
        Creates variables for the training data (since our training data is small) with initialisers
        :param loaded_data: The loaded data returned from vecs.load_from_dir_root.
        :return: training_vars: dict with file_i as keys and list of three variables as value
        feed_dict: dict with placeholders as keys and training data as values
        """

    #TODO: At the moment the batch_size has to be the same as the number of files in loaded_data

    num_files = len(loaded_data)

    assert num_files == batch_size

    # training_vars = dict()
    # training_inits = dict()
    # feed_dict = dict()

    data = []
    max_sines = loaded_data[0][0].shape[1]
    with tf.variable_scope("training_data"):
        for i in range(num_files):
            with tf.variable_scope('file_{}'.format(i)):
                assert len(loaded_data[i]) == 3  # Should be xtfreq, xtmag, xtphase

                xtfreq_mag_phase = np.hstack((loaded_data[i][0], loaded_data[i][1], loaded_data[i][2]))
                data.append(xtfreq_mag_phase)

        zero_start = np.zeros((batch_size, 1, 3 * max_sines))
        data = np.stack(data) # (batch_size, n_frames, 3 * max_sines)
        input_data = np.concatenate((zero_start, data[:, :-1, :]), axis=1) # Zero vector in the beginning, remove the last frame
        output_data = data

        data_dict = {'output_data' : output_data, 'input_data': input_data}
        placeholders = {'output_data' : tf.placeholder(dtype=tf.float32, shape=(output_data.shape)),
                        'input_data': tf.placeholder(dtype=tf.float32, shape=(input_data.shape))}

    return placeholders, data_dict
