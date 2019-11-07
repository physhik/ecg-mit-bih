FROM python:3.6.8-slim-stretch

COPY src/*.py /usr/src/app/src/
COPY requirements.txt /usr/src/app/src/
COPY src/templates/ /usr/src/app/src/templates/
COPY src/static/ /usr/src/app/src/static
COPY src/models/MLII-latest.hdf5 /usr/src/app/src/models/
COPY src/training2017/ /usr/src/app/src/training2017/

WORKDIR /usr/src/app/src/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
