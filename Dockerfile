# Use nvidia/cuda image
FROM --platform=linux/amd64 nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
ARG USE_CUDA=False
ARG VIDEO_URL

# Set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion libgl1 ffmpeg -y && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

WORKDIR /opt/ml/code

# Install dependencies
COPY ./environment.yaml environment.yaml
RUN conda env create -f environment.yaml
ENV PATH /opt/conda/envs/real_time_object_detection/bin:$PATH
ENV CONDA_DEFAULT_ENV real_time_object_detection
RUN /bin/bash -c "source activate real_time_object_detection"

COPY . /opt/ml/code

ENV USE_CUDA $USE_CUDA
ENV VIDEO_URL $VIDEO_URL

EXPOSE 5000

ENTRYPOINT [ "/opt/ml/code/entrypoint.sh" ]