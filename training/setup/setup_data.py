import tensorflow as tf
import numpy as np

def setup_training_data(loaded_data, batch_size):
    """
        Creates variables for the training data (since our training data is small) with initialisers
        :param loaded_data: The loaded data returned from vecs.load_from_dir_root.
        :return: training_vars: dict with file_i as keys and list of three variables as value
        feed_dict: dict with placeholders as keys and training data as values
                    NB placeholders/data shape (n_frames, batch_size, 3*max_sines)
        """

    #TODO: At the moment the batch_size has to be the same as the number of files in loaded_data

    num_files = len(loaded_data)

    assert num_files == batch_size

    data = []
    max_sines = loaded_data[0][0].shape[1]
    with tf.variable_scope("training_data"):
        for i in range(num_files):
            with tf.variable_scope('file_{}'.format(i)):
                assert len(loaded_data[i]) == 4  # Should be xtfreq, xtmag, xtphase, active_tracks

                xtfreq_mag_phase_active = np.hstack((loaded_data[i][0], loaded_data[i][1],
                                                     loaded_data[i][2], loaded_data[i][3]))
                data.append(xtfreq_mag_phase_active)

        zero_start = np.zeros((batch_size, 1, 4 * max_sines))
        data = np.stack(data)  # (batch_size, n_frames, 4 * max_sines)
        input_data = np.concatenate((zero_start, data[:, :-1, :]), axis=1)  # Zero vector in the beginning, remove the last frame
        output_data = data

        output_data = np.transpose(output_data, (1, 0, 2))  # (n_frames, batch_size, 4 * max_sines)
        input_data = np.transpose(input_data, (1, 0, 2))  # (n_frames, batch_size, 4 * max_sines)

        print('Data shapes: ', input_data.shape, output_data.shape)

        data_dict = {'output_data' : output_data, 'input_data': input_data}
        placeholders = {'output_data' : tf.placeholder(dtype=tf.float32, shape=(output_data.shape)),
                        'input_data': tf.placeholder(dtype=tf.float32, shape=(input_data.shape))}

    return placeholders, data_dict

def setup_training_data_for_stft(loaded_data, batch_size):
    """
        Creates variables for the training data (since our training data is small) with initialisers
        :param loaded_data: The loaded data returned from vecs.load_from_dir_root.
        :return: training_vars: dict with file_i as keys and list of three variables as value
        feed_dict: dict with placeholders as keys and training data as values
                    NB placeholders/data shape (n_frames, batch_size, 3*max_sines)
        """

    #TODO: At the moment the batch_size has to be the same as the number of files in loaded_data

    num_files = len(loaded_data)

    assert num_files == batch_size

    data = []
    num_freqs = loaded_data[0][0].shape[1]
    with tf.variable_scope("training_data"):
        for i in range(num_files):
            with tf.variable_scope('file_{}'.format(i)):
                assert len(loaded_data[i]) == 2  # Should be xtmag, xtphase

                xtfreq_mag_phase_active = np.hstack((loaded_data[i][0], loaded_data[i][1]))
                data.append(xtfreq_mag_phase_active)

        zero_start = np.zeros((batch_size, 1, 2 * num_freqs))
        data = np.stack(data)  # (batch_size, n_frames, 4 * max_sines)
        input_data = np.concatenate((zero_start, data[:, :-1, :]), axis=1)  # Zero vector in the beginning, remove the last frame
        output_data = data

        output_data = np.transpose(output_data, (1, 0, 2))  # (n_frames, batch_size, 4 * max_sines)
        input_data = np.transpose(input_data, (1, 0, 2))  # (n_frames, batch_size, 4 * max_sines)

        print('Data shapes: ', input_data.shape, output_data.shape)

        data_dict = {'output_data' : output_data, 'input_data': input_data}
        placeholders = {'output_data' : tf.placeholder(dtype=tf.float32, shape=(output_data.shape)),
                        'input_data': tf.placeholder(dtype=tf.float32, shape=(input_data.shape))}

    return placeholders, data_dict