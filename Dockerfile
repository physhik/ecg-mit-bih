FROM python:3.12.9-slim-bookworm

COPY src/*.py /usr/src/app/src/
COPY requirements.txt /usr/src/app/src/
COPY src/templates/ /usr/src/app/src/templates/
COPY src/static/ /usr/src/app/src/static
COPY src/uploads /usr/src/app/src/uploads/
COPY models/MLII-latest.keras /usr/src/app/src/models/
COPY training2017/ /usr/src/app/src/training2017/

WORKDIR /usr/src/app/src/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

