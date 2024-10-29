FROM jupyter/minimal-notebook:latest

# --- Define Environment Variables--- #

ENV DISPLAY=:100

# --- Install basic tools --- #
USER root
RUN  apt update -q && apt install -y \
        software-properties-common \
        gnupg2 \
        curl \
        git \
        wget \
        vim \
        nano \
        net-tools\
        ca-certificates \
        apt-transport-https \
        build-essential \
        lsb-release \
        binutils

# Set locale
RUN apt update && \
    apt install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# --- Install Oh-my-bash --- #
USER ${NB_USER}
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" --unattended
COPY --chown=${NB_USER}:users ./bashrc.sh /home/${NB_USER}/.bashrc

USER root
# --- Install XPRA and GUI tools --- #
ARG REPOFILE=https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/jammy/xpra.sources
RUN wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    cd /etc/apt/sources.list.d && wget $REPOFILE
RUN apt update && apt install -y \
        xpra \
        gdm3 \
        tmux \
        nautilus \
        gnome-shell \
        gnome-session \
        gnome-terminal \
        libqt5x11extras5 \
        xvfb && \
    apt clean

# --- Install python packages --- #
USER ${NB_USER}
USER root
RUN pip install --upgrade \
        jupyterlab==3.6.6 \
        ipywidgets \
        jupyter-resource-usage \
        jupyter-server-proxy \
        jupyter-ai \
        openai \
        Pillow \
        rosdep \
        lark \
        catkin_tools \
        colcon-common-extensions \
        notebook \
        pandas \
        openpyxl \
        matplotlib \
        seaborn \
        scikit-learn \
        torch \
        torchvision \
        torchtext==0.6.0 \
        nltk \
        transformers \
        flwr \
        tqdm \
        typing \
        xgboost \
        gpytorch \
    && pip cache purge

USER root
# --- Install jupyterlab extensions --- #
COPY --chown=${NB_USER}:users jupyter-extensions /home/${NB_USER}/.jupyter-extensions
RUN pip install -e /home/${NB_USER}/.jupyter-extensions/jupyter-xprahtml5-proxy


USER root
RUN pip install --upgrade "jupyterlab<4" jupyterlab-git


# Quickly install missing packages
USER root


# Install mosquitto and paho mqtt


USER ${NB_USER}
WORKDIR /home/${NB_USER}
# --- Appy JupyterLab custom Settings --- #
COPY --chown=${NB_USER}:users ./jupyter-settings.json /opt/conda/share/jupyter/lab/settings/overrides.json


# --- Entrypoint --- #
COPY --chown=${NB_USER}:users entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD [ "start-notebook.sh" ]

