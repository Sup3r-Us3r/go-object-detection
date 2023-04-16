FROM golang:1.20.1

RUN groupadd wheel  && \
  groupadd storage && \
  groupadd power

RUN useradd -m -g users -G wheel,storage,power,video -s /bin/bash appuser && \
  echo 'root:toor' | chpasswd && \
  echo 'appuser:toor' | chpasswd

# Update system
RUN apt-get update

# Install sudo
RUN apt-get -y install sudo

# Allows running commands as root
RUN echo 'appuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install dependency to use Webcam
RUN apt-get -y install v4l-utils

# Install dependency to handle X display server (X11)
RUN apt-get -y install xauth

# Install the protocol buffers library and compiler
RUN apt-get -y install --no-install-recommends \
  libprotobuf-dev \
  protobuf-compiler

# Install Tensorflow
RUN wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz && \
  tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz && \
  rm -rf libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz && \
  ldconfig /usr/local/lib

# Install OpenCV
RUN git clone https://github.com/hybridgroup/gocv.git && \
  cd gocv && \
  make install && \
  cd .. && \
  rm -rf gocv

ENV GOPATH /go

WORKDIR $GOPATH/app

RUN chown -R 1000:1000 $GOPATH

RUN ln -sf /usr/local/go/bin/go /usr/bin/go

USER appuser

ENTRYPOINT [ "tail", "-f", "/dev/null" ]
