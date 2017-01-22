from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
import os
from nn_models.vae_encoders import LSTMEncoder
from nn_models.vae_decoders import LSTMDecoder
from nn_models.vae import VAE
import tensorflow as tf
from utils.vectorisation_utils import create_json
from datetime import datetime
import json

def main(args):
    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)

    model_folder = args.model_folder

    deterministic_warm_up = args.deterministic_warm_up
    batch_size = args.batch_size
    samples_per_batch = args.samples_per_batch
    dataset = DatasetFeed(loaded, batch_size)

    dataset.set_all_data_blocks_to_max_shape()  # Creates the dataset.data_masks list
    data_shape = dataset.max_data_shape

    n_steps = data_shape[0]
    n_outputs = data_shape[1]

    if os.path.exists(model_folder):  # Resume a previously trained model
        if not os.path.isfile(model_folder + ".meta"):
            raise ValueError("model_folder exists, but supplied path is not a meta file (supply path without '.meta'")
        print("Model folder exists. Resuming training from {}".format(model_folder))
        meta_graph = model_folder
        model_folder = '/'.join(meta_graph.split('/')[:-1]) + '/'
        with open(model_folder + '/network_settings.json') as json_file:
            json_settings = json.load(json_file)
        n_input = json_settings['n_inputs']
        assert n_outputs == json_settings['n_outputs'], "Vector outputs size does not match that stored" \
                                                                     "in the network_settings.json file"
        assert n_steps == json_settings['n_steps'], "Vector n_steps does not match that stored in the " \
                                                                 "network_settings.json file"
        log_dir = model_folder + 'log/'  # For tensorboard
        analysis_dir = model_folder + 'analysis/'  # For learning curves etc.
        for dir in [log_dir, analysis_dir]:
            if not os.path.exists(dir): os.makedirs(dir)

        vae = VAE(model_to_restore=meta_graph, d_hyperparams={'deterministic_warm_up': deterministic_warm_up,
                                                              'samples_per_batch': samples_per_batch},
                  log_dir=log_dir, json_dict=json_settings)
        vae.dataset = dataset

        cost = vae.train(max_iter=args.num_training_steps)
        json_settings['epochs_completed'] = vae.dataset.epochs_completed
        json_settings['cost'] = float(cost)

        create_json(model_folder + 'network_settings.json', json_settings)
    else:  # Create a new model
        if model_folder[-1] != '/':
            model_folder += '/'
        print("Model folder does not exist, training new model.")
        os.makedirs(model_folder)

        latent_dim = args.latent_space_dimension
        n_hidden_encoder = args.lstm_encoder_hidden_units[0]  # Just one hidden layer for now
        prelatent_dense_layers = args.prelatent_dense_layers
        n_hidden_decoder = args.lstm_decoder_hidden_units[0]
        postlatent_dense_layers = args.postlatent_dense_layers

        encoder = LSTMEncoder(n_hidden_encoder, prelatent_dense_layers, latent_dim)
        decoder = LSTMDecoder(n_hidden_decoder, postlatent_dense_layers, n_outputs, n_steps, output_activation=tf.sigmoid)

        n_input = n_outputs

        model_datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        json_settings = {'model_datetime': model_datetime,
                         'n_hidden_encoder': n_hidden_encoder,
                         'n_hidden_decoder': n_hidden_decoder,
                         'prelatent_dense_layers': prelatent_dense_layers,
                         'poslatent_dense_layers': postlatent_dense_layers,
                         'n_outputs': n_outputs,
                         'n_inputs': n_input,
                         'n_steps': n_steps,
                         analysis_type + '_settings': json_vector_settings,
                         'analysis_type': analysis_type,
                         'latent_dim': latent_dim}

        input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size * samples_per_batch, n_input],
                                           name="input")
        # The following placeholder is for feeding the ground truth to the LSTM decoder - the first input should be zeros
        shifted_input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size * samples_per_batch, n_input],
                                                   name="shifted_input")
        print('input_placeholder shape: ', input_placeholder.get_shape())

        log_dir = model_folder + 'log/'  # For tensorboard
        analysis_dir = model_folder + 'analysis/'  # For learning curves etc.
        for dir in [log_dir, analysis_dir]:
            if not os.path.exists(dir): os.makedirs(dir)

        build_dict = {'encoder': encoder,
                      'decoder': decoder,
                      'n_input': n_input,
                      'input_placeholder': input_placeholder,
                      'shifted_input_placeholder': shifted_input_placeholder,
                      'latent_dim': latent_dim,
                      'dataset': dataset,
                      'model_folder': model_folder}

        vae = VAE(build_dict=build_dict,
                  d_hyperparams={'deterministic_warm_up': deterministic_warm_up,
                                 'samples_per_batch': samples_per_batch},
                  log_dir=log_dir, json_dict=json_settings)

        create_json(model_folder + 'network_settings.json', json_settings)

        cost = vae.train(max_iter=args.num_training_steps)
        json_settings['epochs_completed'] = vae.dataset.epochs_completed
        json_settings['cost'] = float(cost)
        json_settings['global_step'] = int(vae.step)

        create_json(model_folder + 'network_settings.json', json_settings)


