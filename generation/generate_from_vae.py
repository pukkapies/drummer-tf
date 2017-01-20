import os
from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
from nn_models.vae import VAE
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.generation_utils import SineModelOutputProcessing, SineModelOutputProcessingWithActiveTracking
from utils.generation_utils import STFTModelOutputProcessing
from models.utilFunctions import nextbiggestpower2, window_dictionary
from models.sineModel import sineModelSynth
from models.stft import stftSynth, stftAnal
from plotting import spectogram_plot
from datetime import datetime
import soundfile


STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

def main(args):
    meta_graph = args.model_folder

    model_folder = '/'.join(meta_graph.split('/')[:-1]) + '/'
    model_name = os.path.basename(meta_graph)
    plot_folder = args.plot_folder + model_name
    if not os.path.exists(plot_folder): os.makedirs(plot_folder)

    with open(model_folder + 'network_settings.json', 'r') as f:
        network_settings = json.load(f)

    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)
    dataset = DatasetFeed(loaded, 9)
    # for datablock in dataset.data:
    #     print(datablock.shape)

    dataset.set_all_data_blocks_to_max_shape(json_vector_settings['mag_normalised_range'][0])
    # data_shape = dataset.max_data_shape

    # n_steps = data_shape[0]
    # n_outputs = data_shape[1]


    #TODO: Retrieve these from a json
    # n_hidden_encoder = 700
    # n_hidden_decoder = 700
    # batch_size = 96
    # latent_dim = 2
    # n_input = n_outputs

    # encoder = LSTMEncoder(n_hidden_encoder, latent_dim)
    # decoder = LSTMDecoder(n_hidden_decoder, n_outputs, n_steps, output_activation=tf.sigmoid)
    #
    # input_placeholder = tf.placeholder(tf.float32, shape=[n_steps, batch_size, n_input], name="x")

    vae = VAE(model_to_restore=meta_graph)

    # HANDLES
    # vae.x_in, vae.z_mean, vae.z_log_sigma,
    # vae.x_reconstructed, vae.z_, vae.x_reconstructed_,
    # vae.cost, vae.global_step, vae.train_op

    ##################### PLOT IN LATENT SPACE #####################

    mus, _ = vae.encode(np.transpose(dataset.next_batch(), [1, 0, 2]))  # (n_steps, batch_size, n_inputs)
    ys, xs = mus.T

    print('Means of z variable:', mus)

    plt.figure()
    plt.title("round {}: {} in latent space".format(vae.step, 'Toms'))
    kwargs = {'alpha': 0.8}

    labels = [0]*3 + [1]*3 + [2]*3  # Total hack to label different classes of audio files and mark them on the plot
    #TODO: Store classes in the dataset (maybe with filenames)
    classes = set(labels)
    if classes:
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs['c'] = [colormap[i] for i in labels]

        # make room for legend
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                    for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                    fancybox=True, loc='center left')

    plt.scatter(xs, ys, **kwargs)

    # if range_:
    #     plt.xlim(*range_)
    #     plt.ylim(*range_)

    # plt.show()
    title = "latent_space.png"
    plt.savefig(os.path.join(plot_folder, title), bbox_inches="tight")

    ##################### EXPLORE LATENT SPACE #####################

    min_, max_, nx, ny = -4, 4, 5, 5

    # complex number steps act like np.linspace
    # row, col indices (i, j) correspond to graph coords (y, x)
    # rollaxis enables iteration over latent space 2-tuples
    zs = np.rollaxis(np.mgrid[max_:min_:ny * 1j, min_:max_:nx * 1j], 0, 3)

    M = json_vector_settings['M']
    N = json_vector_settings['N']
    H = json_vector_settings['H']
    sr = json_vector_settings['sample_rate']
    analysis_settings = network_settings['stft_settings']

    for zrow in zs:
        # zrow is a matrix, which will be interpreted as (batch_size, 2)
        for z in zrow:
            z = np.expand_dims(z, axis=0)  # TODO: Allow batches for z decoder
            generation = vae.decode(z)  # shape (n_steps, batch_size, n_outputs) -> batch_size = 1
            generation = np.squeeze(generation, axis=1)  # shape (n_steps, n_outputs)
            # Plot and generate audio
            if analysis_type == 'sine_model':
                process_output = SineModelOutputProcessingWithActiveTracking(generation, network_settings)
                xtfreq, xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
                reconstruction = sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(analysis_settings['M']), H,
                                                sr)
            elif analysis_type == 'stft':
                process_output = STFTModelOutputProcessing(generation, network_settings)
                xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
                np.save(plot_folder + 'xtmag_model', xtmag)
                np.save(plot_folder + 'xtphase_model', xtphase)
                plot_filepath = './generation/plots/VAE/{}-(generated_{})'.format(model_name, STARTED_DATESTRING)
                if not os.path.exists(plot_filepath): os.makedirs(plot_filepath)
                spectogram_plot(xtmag, xtphase, M, N, H, sr, show=False,
                                filepath=plot_filepath + '/network_output_spectogram_{}'.format(z))
                reconstruction = stftSynth(xtmag, xtphase, M, H)
            else:
                raise Exception('analysis_type not recognised!')

            soundfile.write('./generation/wav_output/VAE/{}-{}-(generated_{}).wav'.format(model_name, z, STARTED_DATESTRING),
                            reconstruction, sr, format='wav')




    ###############################################################