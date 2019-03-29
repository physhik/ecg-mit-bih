from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from graph import ECG_model
from config import get_config
from utils import *

def train(config, X, y, Xval=None, yval=None):
    
    classes = ['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
    Xe = np.expand_dims(X, axis=2)
    if not config.split:
        from sklearn.model_selection import train_test_split
        Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.2, random_state=1)
    else:
        Xvale = np.expand_dims(Xval, axis=2)
        (m, n) = y.shape
        y = y.reshape((m, 1, n ))
        (mvl, nvl) = yval.shape
        yval = yval.reshape((mvl, 1, nvl))

    if config.checkpoint_path is not None:
        model = model.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch # put the resuming epoch
    else:
        model = ECG_model(config)
        initial_epoch = 0

    mkdir_recursive('models')
    #lr_decay_callback = LearningRateSchedulerPerBatch(lambda epoch: 0.1)
    callbacks = [
            EarlyStopping(patience = config.patience, verbose=1),
            ReduceLROnPlateau(factor = 0.5, patience = 3, min_lr = 0.01, verbose=1),
            TensorBoard( log_dir='./logs', histogram_freq=0, write_graph = True, write_grads=False, write_images=True),
            ModelCheckpoint('models/{}-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, period=10)
            # , lr_decay_callback
    ]

    model.fit(Xe, y,
            validation_data=(Xvale, yval),
            epochs=config.epochs,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)
    print_results(config, model, Xvale, yval, classes, )

    #return model

def main(config):
    print('feature:', config.feature)
    #np.random.seed(0)
    (X,y, Xval, yval) = loaddata(config.input_size, config.feature)
    train(config, X, y, Xval, yval)

if __name__=="__main__":
    config = get_config()
    main(config)
