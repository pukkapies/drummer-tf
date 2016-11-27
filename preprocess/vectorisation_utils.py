import numpy as np
import os
import json

filenames_list_dict = {'stft': ['mag.npy', 'phase.npy'],
                       'sine_model': ['freq.npy', 'mag.npy', 'phase.npy', 'active_tracks.npy']}


class InvalidPathError(Exception): pass

def load_npy(filepath, filenames_list):
    if not os.path.exists(filepath):
        raise InvalidPathError("{} does not exist!".format(filepath))
    data = []
    for i in range(len(filenames_list)):
        data.append(np.load(filepath + '/' + filenames_list[i]))
    return data

def load_from_dir_root(rootdir, analysis_type):
    """
    Loads all data saved in a given folder. Searches through all subfolders and finds every folder that contains
    the file names listed in 'filenames'. Returns them in the list 'loaded_data'. Also searches for the
    vectorisation settings file according to 'analysis_type' and returns a dict with the settings.
    :param rootdir: Path to the root directory where all data is saved
    :param filenames: List of filenames that comprise the data, e.g. ['mag.npy', 'phase.npy']
    :param analysis_type: String, e.g. 'stft' or 'sine_model'
    :return: loaded_data list of lists, dict of vectorisation settings
    """
    filenames_list = filenames_list_dict['analysis_type']
    if not os.path.exists(rootdir):
        raise InvalidPathError("{} does not exist!".format(rootdir))
    if not os.path.exists(rootdir + '/{}_settings.json'.format(analysis_type)):
        raise Exception("{}_settings.json file not found. Maybe the data hasn't been vectorised yet.".format(analysis_type))
    with open(rootdir + '/{}_settings.json'.format(analysis_type)) as json_file:
        json_vector_settings_dict = json.load(json_file)
    loaded_data = []
    for root, dir, filenames in os.walk(rootdir):
        if all(x in filenames for x in filenames_list):
            print('Loading files from {}...'.format(root), end='')
            data = load_npy(root, filenames_list)
            print(data)
            print('done')
            loaded_data.append(data)
    return loaded_data, json_vector_settings_dict

def create_json(settings_file, json_dict):
    with open(settings_file, 'w') as json_file:
        json.dump(json_dict, json_file)