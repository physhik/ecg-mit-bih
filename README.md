[![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](./LICENSE)

## ECG classification using MIT-BIH dataset 

This repo is an implementation of https://www.nature.com/articles/s41591-018-0268-3 and https://arxiv.org/abs/1707.01836

and focus on training using a MIT-BIH dataset. If you want to train using CINC or open irhythm data, see the open source which the authors of the original research paper have coded at https://github.com/awni/ecg

Introduction to MIT-BIH dataset at physionet : https://physionet.org/physiobank/database/mitdb/

### Dependency 

Consistent with the environment of Google colab with wfdb, deepdish installations and numpy reinstallation. 

- Python >= 3.6.7
- keras== 2.2.5 
- scikit-learn==0.21.3
- wfdb==2.2.1
- deepdish==0.3.6
- scipy==1.3.1
- numpy==1.15.4
- tqdm==4.28.1
- six==1.12.0

I recommend using a vitual enviroment for Python. 

### Data setup and train 

```
$ git clone https://github.com/physhik/ecg-mit-bih.git
$ cd ecg-mit-bih
$ pip install -r requirements.txt
$ cd src
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


### Jupyter notebook example

In case, you do not have a GPU above a decent performance, you might be able to use Google colab. Follow the [Jupyter notebook](https://github.com/physhik/ecg-mit-bih/blob/master/src/practice/ecg_mit.ipynb).

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
