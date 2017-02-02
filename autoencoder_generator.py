import numpy as np
import argparse
import os
from utils.vectorisation_utils import load_from_dir_root
from utils.dataset import DatasetFeed
from nn_models.autoencoder import Autoencoder
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

SAMPLES = 16000
MODEL_FOLDER = './training/saved_models/'
SAVE_EVERY = None
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
WAV_OUT_PATH = './generation-' + STARTED_DATESTRING
PLOT_FOLDER = './generation/plots/AE/'


def main(args):
    meta_graph = args.model_folder

    model_folder = '/'.join(meta_graph.split('/')[:-1]) + '/'
    model_name = os.path.basename(meta_graph)
    plot_folder = args.plot_folder + model_name
    if not os.path.exists(plot_folder): os.makedirs(plot_folder)

    with open(model_folder + 'network_settings.json', 'r') as f:
        network_settings = json.load(f)

    loaded, json_vector_settings, analysis_type = load_from_dir_root(args.vector_folder)
    dataset = DatasetFeed(loaded, 15)
    # for datablock in dataset.data:
    #     print(datablock.shape)

    dataset.set_all_data_blocks_to_max_shape(json_vector_settings['mag_normalised_range'][0])
    # data_shape = dataset.max_data_shape

    # n_steps = data_shape[0]
    # n_outputs = data_shape[1]


    #TODO: Retrieve these from a json
    # batch_size = 96
    # latent_dim = 2
    # n_input = n_outputs

    autoencoder = Autoencoder(model_to_restore=meta_graph)
    global_step = (meta_graph.split("/")[-1]).split('-')[-1]

    # HANDLES
    # autoencoder.input_placeholder, autoencoder.shifted_input_placeholder,
    # autoencoder.encoding_cell, autoencoder.encoding_hidden, autoencoder.x_reconstructed, autoencoder.encoding_cell,
    # autoencoder.encoding_hidden_, autoencoder.x_reconstructed_, autoencoder.cost, autoencoder.rec_loss,
    # autoencoder.l2_reg, autoencoder.apply_gradients_op, autoencoder.global_step

    ##################### PLOT IN LATENT SPACE #####################

    next_batch = dataset.next_batch()  # (batch_size, n_steps, n_inputs)
    next_batch_original = np.transpose(next_batch, [1, 0, 2])  # (n_steps, batch_size, n_inputs)
    next_batch = next_batch_original[::-1, :, :]  # Reverse the data in time!!

    lstm_state_tuple = autoencoder.encode(next_batch)  # (n_steps, batch_size, n_inputs)
    c, h = lstm_state_tuple

    print("c shape: {}, h shape: {}".format(c.shape, h.shape))
    print(c)
    print(h)

    xs, ys = c.T[:2, :]

    plt.figure()
    plt.title("round {}: {} in latent space".format(global_step, 'Toms'))
    kwargs = {'alpha': 0.8}

    labels = [0]*32 + [1]*32 + [2]*32  # Total hack to label different classes of audio files and mark them on the plot
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

    # plt.show()
    title = "latent_space.png"
    plt.savefig(os.path.join(plot_folder, title), bbox_inches="tight")

    ####################### TEST RECONSTRUCTION ######################

    reconstructions = autoencoder.ae(next_batch)

    estimated_cost = np.mean((reconstructions - next_batch_original))
    print("Estimated cost: {}".format(estimated_cost))


    # ##################### EXPLORE LATENT SPACE #####################
    #
    # min_, max_, nx, ny = -4, 4, 5, 5
    #
    # # complex number steps act like np.linspace
    # # row, col indices (i, j) correspond to graph coords (y, x)
    # # rollaxis enables iteration over latent space 2-tuples
    # zs = np.rollaxis(np.mgrid[max_:min_:ny * 1j, min_:max_:nx * 1j], 0, 3)
    #
    # M = json_vector_settings['M']
    # N = json_vector_settings['N']
    # H = json_vector_settings['H']
    # sr = json_vector_settings['sample_rate']
    # analysis_settings = network_settings['stft_settings']
    #
    # for zrow in zs:
    #     # zrow is a matrix, which will be interpreted as (batch_size, 2)
    #     for z in zrow:
    #         z = np.expand_dims(z, axis=0)  # TODO: Allow batches for z decoder
    #         generation = vae.decode(z)  # shape (n_steps, batch_size, n_outputs) -> batch_size = 1
    #         generation = np.squeeze(generation, axis=1)  # shape (n_steps, n_outputs)
    #         # Plot and generate audio
    #         if analysis_type == 'sine_model':
    #             process_output = SineModelOutputProcessingWithActiveTracking(generation, network_settings)
    #             xtfreq, xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
    #             reconstruction = sineModelSynth(xtfreq, xtmag, xtphase, nextbiggestpower2(analysis_settings['M']), H,
    #                                             sr)
    #         elif analysis_type == 'stft':
    #             process_output = STFTModelOutputProcessing(generation, network_settings)
    #             xtmag, xtphase = process_output.convert_network_output_to_analysis_model_input()
    #             np.save(plot_folder + 'xtmag_model', xtmag)
    #             np.save(plot_folder + 'xtphase_model', xtphase)
    #             plot_filepath = './generation/plots/VAE/{}-(generated_{})'.format(model_name, STARTED_DATESTRING)
    #             if not os.path.exists(plot_filepath): os.makedirs(plot_filepath)
    #             spectogram_plot(xtmag, xtphase, M, N, H, sr, show=False,
    #                             filepath=plot_filepath + '/network_output_spectogram_{}'.format(z))
    #             reconstruction = stftSynth(xtmag, xtphase, M, H)
    #         else:
    #             raise Exception('analysis_type not recognised!')
    #
    #         soundfile.write('./generation/wav_output/VAE/{}-{}-(generated_{}).wav'.format(model_name, z, STARTED_DATESTRING),
    #                         reconstruction, sr, format='wav')




    ###############################################################



def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='VAE model generation script')
    parser.add_argument('--model_folder', type=str,
                        help='Which model checkpoint to generate from (provide filename, without .meta)')
    parser.add_argument('--wav_out_path', type=str, default=WAV_OUT_PATH, help='Path to output wav file')
    parser.add_argument('--vector_folder', type=str, default=None, help='Path to vector folder')
    parser.add_argument('--plot_folder', type=str, default=PLOT_FOLDER, help='Folder to store plots')
    # parser.add_argument(
    #     '--wav_seed',
    #     type=str,
    #     default=None,
    #     help='The wav file to start generation from')
    return parser.parse_args()


if __name__ == '__main__':
    clargs = get_arguments()
    if clargs.plot_folder != '/':
        clargs.plot_folder += '/'

    main(clargs)
