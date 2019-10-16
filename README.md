[![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](./LICENSE)

## ECG classification using MIT-BIH dataset 

This repo is an implementation of https://www.nature.com/articles/s41591-018-0268-3 and https://arxiv.org/abs/1707.01836

and focus on training using a MIT-BIH dataset. If you want to train using CINC or open irhythm data, see the open source which the authors of the original research paper have coded at https://github.com/awni/ecg

Introduction to MIT-BIH dataset at physionet : https://physionet.org/physiobank/database/mitdb/

### Dependency 

Consistent with the environment of Google colab with wfdb, deepdish installations. 

- Python >= 3.6.7
- keras
- scikit-learn
- wfdb
- deepdish
- scipy
- numpy==1.15.4
- tqdm

I recommend using a vitual enviroment for Python. 

### Data setup and train 

```
$ git clone https://github.com/physhik/ecg-mit-bih.git
$ cd ecg-mit-bih/src
$ python data.py --downloading True
$ python train.py
```

### Test

Predict an annotation of [CINC2017 data](https://physionet.org/challenge/2017/) or your own data(csv file)

It randomly chooses one of data, and predict the slices of the signal.

```
$ python predict.py --cinc_download True
```
--cinc_download branch is used at first to download the CINC2017 data.

See config.py and customize your parameters or learn better way to train and test 


### Introduction to ECG 

I presented a bit more about ECG classfications on my personal blog, http://physhik.com

### Reference to 

The original research papers
https://www.nature.com/articles/s41591-018-0268-3
https://arxiv.org/abs/1707.01836

The open source by authors
https://github.com/awni/ecg

also noticable 
https://github.com/fernandoandreotti/cinc-challenge2017/tree/master/deeplearn-approach
