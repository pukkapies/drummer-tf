from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
import os
from nn_models.vae_encoders import LSTMEncoder
from nn_models.vae_decoders import LSTMDecoder
from nn_models.vae import VAE
import tensorflow as tf
from utils.vectorisation_utils import create_json

def main(args):
    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)

    model_folder = args.model_folder
    if model_folder[-1] != '/':
        model_folder += '/'

    log_dir = model_folder + 'log/'  # For tensorboard

    batch_size = args.batch_size
    dataset = DatasetFeed(loaded, batch_size)

    #TODO Figure out a better way to choose the padding value
    dataset.set_all_data_blocks_to_max_shape(json_vector_settings['mag_normalised_range'][0])
    data_shape = dataset.max_data_shape

    n_steps = data_shape[0]
    n_outputs = data_shape[1]

    if not os.path.exists(model_folder):
        print("Model folder does not exist, training new model.")
        os.makedirs(model_folder)
    else:
        print("Model folder exists. Resuming training not yet supported")
        exit(1)

    latent_dim = args.latent_space_dimension
    n_hidden_encoder = args.lstm_encoder_hidden_units[0]  # Just one hidden layer for now
    prelatent_dense_layers = args.prelatent_dense_layers
    n_hidden_decoder = args.lstm_decoder_hidden_units[0]
    postlatent_dense_layers = args.postlatent_dense_layers

    encoder = LSTMEncoder(n_hidden_encoder, prelatent_dense_layers, latent_dim)
    decoder = LSTMDecoder(n_hidden_decoder, postlatent_dense_layers, n_outputs, n_steps, output_activation=tf.sigmoid)

    n_input = n_outputs

    json_settings = {'n_hidden_encoder': n_hidden_encoder,
                     'n_hidden_decoder': n_hidden_decoder,
                     'prelatent_dense_layers': prelatent_dense_layers,
                     'poslatent_dense_layers': postlatent_dense_layers,
                     'n_outputs': n_outputs,
                     'n_inputs': n_input,
                     'n_steps': n_steps,
                     analysis_type + '_settings': json_vector_settings,
                     'analysis_type': analysis_type,
                     'latent_dim': latent_dim}

    input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size, n_input], name="x")
    print('input_placeholder shape: ', input_placeholder.get_shape())

    build_dict = {'encoder': encoder, 'decoder': decoder, 'n_input': n_input, 'input_placeholder': input_placeholder,
                  'latent_dim': latent_dim, 'dataset': dataset, 'model_folder': model_folder}

    vae = VAE(build_dict=build_dict, d_hyperparams={'KL_loss_coeff': 0.}, log_dir=log_dir)

    create_json(model_folder + 'network_settings.json', json_settings)

    cost = vae.train(max_iter=args.num_training_steps)
    json_settings['epochs_completed'] = vae.dataset.epochs_completed
    json_settings['cost'] = float(cost)

    create_json(model_folder + 'network_settings.json', json_settings)

