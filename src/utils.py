from __future__ import division, print_function
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report
import os

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)

class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """ code from https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
    Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1


def plot_confusion_matrix(y_true, y_pred, classes, feature,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """Modification from code at https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"""
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    mkdir_recursive('results')
    fig.savefig('results/confusionMatrix-'+feature+'.eps', format='eps', dpi=1000)
    return ax


# Precision-Recall curves and ROC curves for each class
def PR_ROC_curves(ytrue, ypred, classes, ypred_mat):
    ybool = ypred == ytrue
    f, ax = plt.subplots(3,4,figsize=(10, 10))
    ax = [a for i in ax for a in i]

    e = -1
    for c in classes:
        idx1 = [n for n,x in enumerate(ytrue) if classes[x]==c]
        idx2 = [n for n,x in enumerate(ypred) if classes[x]==c]
        idx = idx1+idx2
        if idx == []:
            continue
        bi_ytrue = ytrue[idx]
        bi_prob = ypred_mat[idx, :]
        bi_ybool = np.array(ybool[idx])
        bi_yscore = np.array([bi_prob[x][bi_ytrue[x]] for x in range(len(idx))])
        try:
            print("AUC for {}: {}".format(c, roc_auc_score(bi_ybool+0, bi_yscore)))
            e+=1
        except ValueError:
            continue
        ppvs, senss, thresholds = precision_recall_curve(bi_ybool, bi_yscore)
        cax = ax[2*e]
        cax.plot(ppvs, senss, lw=2, label="Model")
        cax.set_xlim(-0.008, 1.05)
        cax.set_ylim(0.0, 1.05)
        cax.set_title("Class {}".format(c))
        cax.set_xlabel('Sensitivity (Recall)')
        cax.set_ylabel('PPV (Precision)')
        cax.legend(loc=3)

        fpr, tpr, thresholds = roc_curve(bi_ybool, bi_yscore)
        cax2 = ax[2*e+1]
        cax2.plot(fpr, tpr, lw=2, label="Model")
        cax2.set_xlim(-0.1, 1.)
        cax2.set_ylim(0.0, 1.05)
        cax2.set_title("Class {}".format(c))
        cax2.set_xlabel('1 - Specificity')
        cax2.set_ylabel('Sensitivity')
        cax2.legend(loc=4)

    mkdir_recursive("results")
    plt.savefig("results/model_prec_recall_and_roc.eps",
        dpi=400,
        format='eps',
        bbox_inches='tight')
    plt.close()

def print_results(config, model, Xval, yval, classes):
    model2 = model
    if config.trained_model:
        model.load_weights(config.trained_model)
    else:    
        model.load_weights('models/{}-latest.hdf5'.format(config.feature))
    # to combine different trained models. On testing  
    if config.ensemble:
        model2.load_weight('models/weights-V1.hdf5')
        ypred_mat = (model.predict(Xval) + model2.predict(Xval))/2
    else:
        ypred_mat = model.predict(Xval)  
    ypred_mat = ypred_mat[:,0]
    yval = yval[:,0]

    ytrue = np.argmax(yval,axis=1)
    yscore = np.array([ypred_mat[x][ytrue[x]] for x in range(len(yval))])
    ypred = np.argmax(ypred_mat, axis=1)
    print(classification_report(ytrue, ypred))
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False)
    print("F1 score:", f1_score(ytrue, ypred, average=None))
    PR_ROC_curves(ytrue, ypred, classes, ypred_mat)

def add_noise(config):
    noises = dict()
    noises["trainset"] = list()
    noises["testset"] = list() 
    import csv
    try:
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv')))
    except:
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")
        testlabel = list(csv.reader(open('training2017/REFERENCE.csv')))
    for i, label in enumerate(testlabel):
      if label[1] == '~':
        filename = 'training2017/'+ label[0] + '.mat'
        from scipy.io import loadmat
        noise = loadmat(filename)
        noise = noise['val']
        _, size = noise.shape
        noise = noise.reshape(size,)
        noise = np.nan_to_num(noise) # removing NaNs and Infs
        from scipy.signal import resample
        noise= resample(noise, int(len(noise) * 360 / 300) ) # resample to match the data sampling rate 360(mit), 300(cinc)
        from sklearn import preprocessing
        noise = preprocessing.scale(noise)
        noise = noise/1000*6 # rough normalize, to be improved 
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(noise, distance=150)
        choices = 10 # 256*10 from 9000
        picked_peaks = np.random.choice(peaks, choices, replace=False)
        for j, peak in enumerate(picked_peaks):
          if peak > config.input_size//2 and peak < len(noise) - config.input_size//2:
              start,end  = peak-config.input_size//2, peak+config.input_size//2
              if i > len(testlabel)/6:
                noises["trainset"].append(noise[start:end].tolist())
              else:
                noises["testset"].append(noise[start:end].tolist())
    return noises

def preprocess(data, config):
    sr = config.sample_rate
    if sr == None:
      sr = 300
    data = np.nan_to_num(data) # removing NaNs and Infs
    from scipy.signal import resample
    data = resample(data, int(len(data) * 360 / sr) ) # resample to match the data sampling rate 360(mit), 300(cinc)
    from sklearn import preprocessing
    data = preprocessing.scale(data)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(data, distance=150)
    data = data.reshape(1,len(data))
    data = np.expand_dims(data, axis=2) # required by Keras
    return data, peaks

# predict 
def uploadedData(filename, csvbool = True):
    if csvbool:
      csvlist = list()
      with open(filename, 'r') as csvfile:
        for e in csvfile:
          if len(e.split()) == 1 :
            csvlist.append(float(e))
          else:
            csvlist.append(e)
    return csvlist
