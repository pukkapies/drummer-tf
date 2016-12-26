from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
import os
from nn_models.vae_encoders import LSTMEncoder
from nn_models.vae_decoders import LSTMDecoder
from nn_models.vae import VAE
import tensorflow as tf

def main(args):
    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)

    dataset = DatasetFeed(loaded, 4)

    #TODO Figure out a better way to choose the padding value
    dataset.set_all_data_blocks_to_max_shape(json_vector_settings['mag_normalised_range'][0])
    data_shape = dataset.max_data_shape

    n_steps = data_shape[0]
    n_outputs = data_shape[1]

    if not os.path.exists(args.model_folder):
        print("Model folder does not exist, training new model.")
        os.makedirs(args.model_folder)

    batch_size = args.batch_size
    latent_dim = args.latent_space_dimension
    n_hidden_encoder = args.lstm_encoder_hidden_units[0]  # Just one hidden layer for now
    n_hidden_decoder = args.lstm_decoder_hidden_units[0]

    encoder = LSTMEncoder(n_hidden_encoder, latent_dim)
    decoder = LSTMDecoder(n_hidden_decoder, n_outputs, n_steps, output_activation=tf.sigmoid)

    n_input = n_outputs

    json_settings = {'n_hidden_encoder': n_hidden_encoder,
                     'n_hidden_decoder': n_hidden_decoder,
                     'n_outputs': n_outputs,
                     'n_inputs': n_input,
                     'n_steps': n_steps,
                     analysis_type + '_settings': json_vector_settings,
                     'analysis_type': analysis_type}

    input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size, n_input], name="x")
    print('input_placeholder shape: ', input_placeholder.get_shape())

    vae = VAE(encoder, decoder, n_input, input_placeholder, latent_dim, dataset)

    vae.train()

