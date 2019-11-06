#!/bin/sh

pip install virtualenv
virtualenv -p python flaskapp
source ./flaskapp/bin/activate

pip install -r requirements.txt

./flaskapp/bin/python data.py --downloading True
./flaskapp/bin/python train.py


