import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

PLOT_FOLDER = './generation/plots/'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='LSTM audio synth model')
    parser.add_argument('model_folder', type=str,
                        help='Folder where analysis files are saved.')
    parser.add_argument('-l', '--learning_curves', action="store_true",
                        help='Plot learning curves for minibatch cost at each iteration.')
    parser.add_argument('-s', '--show', action="store_true",
                        help='Show plot (true) or save plot (false), default is True.')
    parser.add_argument('--reconst_cost', action="store_true",
                        help='Whether to plot the reconstruction cost separately.')
    return parser.parse_args()

def get_paths_and_names(model_folder):
    folder_split = model_folder.split('/')
    model_name = folder_split[-2]
    model_type = folder_split[-3]
    print('Extracted model name as {}'.format(model_name), end='')
    print(' and model type as {}'.format(model_type))

    plot_folder = PLOT_FOLDER + model_type + '/' + model_name + '/'
    return model_name, model_type, plot_folder

def plot_total_cost_learning_curve(analysis_folder, model_name, model_type, plot_folder, show):
    total_cost = np.load(analysis_folder + 'total_cost.npy')
    print('Plotting total cost learning curve...', end='')
    plt.figure()
    plt.plot(total_cost)
    plt.title('{} minibatch total cost for {}'.format(model_type, model_name))
    plt.xlabel('Iteration')
    plt.ylabel('Total cost')
    if show:
        plt.show()
    else:
        plt.savefig(plot_folder + 'total_cost_learning_curves.png')
        plt.close()
    print(' done.')

def plot_reconstruction_cost_learning_curve(analysis_folder, model_name, model_type, plot_folder, show):
    reconstruction_cost = np.load(analysis_folder + 'reconstruction_cost.npy')
    print('Plotting reconstruction cost learning curve...', end='')
    plt.figure()
    plt.plot(reconstruction_cost)
    plt.title('{} minibatch reconstruction cost for {}'.format(model_type, model_name))
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction cost')
    if show:
        plt.show()
    else:
        plt.savefig(plot_folder + 'reconstruction_cost_learning_curves.png')
        plt.close()
    print(' done.')


if __name__ == '__main__':

    args = get_arguments()

    model_folder = args.model_folder
    show_plot = args.show

    if model_folder[-1] != '/':
        model_folder += '/'

    analysis_folder = model_folder + 'analysis/'
    model_name, model_type, plot_folder = get_paths_and_names(model_folder)
    if not os.path.exists(plot_folder): os.makedirs(plot_folder)

    if args.learning_curves:
        plot_total_cost_learning_curve(analysis_folder, model_name, model_type, plot_folder, show_plot)
    if args.reconst_cost:
        plot_reconstruction_cost_learning_curve(analysis_folder, model_name, model_type, plot_folder, show_plot)
