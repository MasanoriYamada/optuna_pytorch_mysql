FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
ARG PYTHON_VERSION=3.6

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get -y install language-pack-ja-base language-pack-ja ibus-mozc

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN pip install -U pip
RUN pip install numpy matplotlib bokeh holoviews pandas tqdm sklearn joblib nose pandas tabulate xgboost lightgbm optuna nose coverage
RUN pip install torch torchvision
# install tensorboardX to /tmp for hparams in tensorboardX
RUN pip install -e git+https://github.com/eisenjulian/tensorboardX.git@add-hparam-support#egg=tensorboardX --src /tmp
RUN pip install tensorflow==1.14.0
RUN pip install future moviepy

# mysqlclient
RUN apt-get install -y libssl-dev
RUN apt-get update
RUN apt-get install -y python3-dev libmysqlclient-dev
RUN pip install mysqlclient
