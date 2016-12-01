import os
import tensorflow as tf

def load_saved_model_to_resume_training(saver, sess, model, is_file=False):
    """
    This function looks in the 'checkpoint' file to get the last saved model and restore it.
    NB: You can also just directly call e.g. saver.restore(sess, 'path_to_folder/model-20'
    :param saver: Saver object
    :param sess: Active session
    :param logdir: Folder path where model is saved
    :param is_file: If True, then the logdir should be a file which will be restored directly
    :return: The training iteration step if a model is found, otherwise None, model_folder
    """
    print("Trying to restore saved checkpoints from {} ...".format(model), end='')

    if is_file:
        saver.restore(sess, model)
        filename = model.split(sep='/')[-1]
        model_folder = os.path.join(*(model.split(sep='/')[:-1]))
        print("Trying to extract global step from filename {}".format(filename))
        global_step = [int(s) for s in str.split() if s.isdigit()]
        if len(global_step)==1:
            global_step = global_step[0]
        else:
            global_step = None
        return global_step, model_folder
    else:
        if 'best_model' in os.listdir(model):
            model_folder = os.path.join(*(model.split(sep='/') + ['best_model/']))
            print("Found best_model folder, trying to restore from here...")
        else:
            if model[-1] != '/':
                model_folder = model + '/'
            else:
                model_folder = model
        ckpt = tf.train.get_checkpoint_state(model_folder)
        if ckpt:
            print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            print("  Global step was: {}".format(global_step))
            print("  Restoring...", end="")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(" Done.")
            return global_step, model_folder
        else:
            print(" No checkpoint found.")
            return None, model_folder

def print_(var, name: str, first_n=5, summarize=5):
    """Util for debugging, by printing values of tf.Variable `var` during training"""
    # (tf.Tensor, str, int, int) -> tf.Tensor
    return tf.Print(var, [var], "{}: ".format(name), first_n=first_n,
                    summarize=summarize)
