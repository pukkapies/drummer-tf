from __future__ import print_function

import numpy as np
import os
import json

# List of filenames that comprise the data
filenames_list_dict = {'stft': ['mag.npy', 'phase.npy'],
                       'sine_model': ['freq.npy', 'mag.npy', 'phase.npy', 'active_tracks.npy']}

class InvalidPathError(Exception): pass

def load_npy(filepath, filenames_list):
    """
    Reads the relevant .npy files in a folder and returns them in a list
    :param filepath: Path to folder containing .npy files
    :param filenames_list: List of .npy filenames, as in the filenames_list_dict above
    :return: List of numpy arrays, one for each of the filenames in the filenames_list
    """
    if not os.path.exists(filepath):
        raise InvalidPathError("{} does not exist!".format(filepath))
    data = []
    for i in range(len(filenames_list)):
        data.append(np.load(filepath + '/' + filenames_list[i]))
    return data

def load_from_dir_root(rootdir):
    """
    Loads all data saved in a given folder. Searches through all subfolders and finds every folder that contains
    the file names listed in 'filenames'. Returns them in the list 'loaded_data'. Also searches for the
    vectorisation settings file according to 'analysis_type' and returns a dict with the settings.
    :param rootdir: Path to the root directory where all data is saved
    :return: loaded_data list of lists containing npy arrays, dict of vectorisation settings, analysis_type
    """
    # Find json file to get the analysis type
    for file in os.listdir(rootdir):
        if '_settings.json' in file:
            analysis_type = file[:-14]
    assert analysis_type in filenames_list_dict.keys()

    with open(rootdir + '/{}_settings.json'.format(analysis_type)) as json_file:
        json_vector_settings_dict = json.load(json_file)

    filenames_list = filenames_list_dict[analysis_type]
    if not os.path.exists(rootdir):
        raise InvalidPathError("{} does not exist!".format(rootdir))
    if not os.path.exists(rootdir + '/{}_settings.json'.format(analysis_type)):
        raise Exception("{}_settings.json file not found. Maybe the data hasn't been vectorised yet.".format(analysis_type))

    loaded_data = []
    for root, dir, filenames in os.walk(rootdir):
        if all(x in filenames for x in filenames_list):
            # print('Loading files from {}...'.format(root), end='')
            data = load_npy(root, filenames_list)
            # print('done')
            loaded_data.append(data)
    return loaded_data, json_vector_settings_dict, analysis_type

def create_json(settings_file, json_dict):
    with open(settings_file, 'w') as json_file:
        json.dump(json_dict, json_file, sort_keys=True, indent=4)