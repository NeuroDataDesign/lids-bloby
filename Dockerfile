FROM ubuntu:16.04

RUN apt-get -y upgrade

RUN apt-get update && apt-get -y install \
  build-essential \
  python3-pip \
  python-all-dev \
  zlib1g-dev \
  libjpeg8-dev \
  libtiff5-dev \
  libfreetype6-dev \
  liblcms2-dev \
  libwebp-dev \
  tcl8.5-dev \
  tk8.5-dev \
  python-tk \
  libhdf5-dev \
  libinsighttoolkit4-dev \
  libfftw3-dev \
  libopenblas-base \
  libopenblas-dev \
  python \
  python-dev \
  git \
  build-essential \
  cmake \
  gcc \
  python3-tk

RUN pip3 install --upgrade pip

RUN pip install ipython[all] jupyter

WORKDIR /work
ADD https://api.github.com/repos/neurodatadesign/bloby/git/refs/heads/master version.json
RUN git clone https://github.com/neurodatadesign/bloby.git /work/bloby --branch master --single-branch
WORKDIR /work/bloby

ENV PYTHONPATH="/work/bloby"

RUN python3 setup.py install

EXPOSE 3000

CMD ["jupyter", "notebook", "--port=3000", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
