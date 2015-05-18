FROM      ubuntu:14.04
MAINTAINER Mateusz Kurek master.mateusz@gmail.com


RUN apt-get update
RUN apt-get install -y \
    build-essential \
    checkinstall \
    flex \
    gcc \
    git \
    libboost-all-dev \
    libqt4-dev \
    libqt4-core \
    libqt4-gui \
    libqt4-network \
    libqt4-opengl \
    libqt4-dev \
    libaudio-dev \
    libxt-dev \
    libpng12-dev \
    libxi-dev \
    libxrender-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libglib2.0-dev \
    pkg-config \
    python \
    qt4-dev-tools \
    wget \
    vim
RUN apt-get install -y python-virtualenv liblapack-dev libatlas-dev gfortran
RUN pip install numpy Theano

RUN mkdir -p /home/soccer
RUN groupadd -r soccer --gid=1000 && useradd -r --uid=1000 --gid=1000 -g soccer --shell /bin/bash --home /home/soccer soccer
RUN chgrp -R soccer /home/soccer && chown -R soccer /home/soccer
RUN adduser soccer sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER soccer

WORKDIR /home/soccer/
RUN wget http://launchpadlibrarian.net/140087283/libbison-dev_2.7.1.dfsg-1_amd64.deb && sudo dpkg -i libbison-dev_2.7.1.dfsg-1_amd64.deb
RUN wget http://launchpadlibrarian.net/140087282/bison_2.7.1.dfsg-1_amd64.deb && sudo dpkg -i bison_2.7.1.dfsg-1_amd64.deb

RUN wget http://heanet.dl.sourceforge.net/project/sserver/rcssserver/15.2.2/rcssserver-15.2.2.tar.gz
RUN wget http://softlayer-ams.dl.sourceforge.net/project/sserver/rcssmonitor/15.1.1/rcssmonitor-15.1.1.tar.gz

RUN tar -xzvf rcssserver-15.2.2.tar.gz
RUN tar -xzvf rcssmonitor-15.1.1.tar.gz

RUN ln -s rcssserver-15.2.2 rcssserver
RUN ln -s rcssmonitor-15.1.1 rcssmonitor_qt4

RUN cd ~/rcssserver && ./configure --with-boost-libdir=/usr/lib/x86_64-linux-gnu/ && make
RUN cd ~/rcssmonitor_qt4 && ./configure --with-boost-libdir=/usr/lib/x86_64-linux-gnu/ && make

#RUN git clone https://github.com/tjpalmer/keepaway.git
RUN sudo cp /root/.bashrc /home/soccer/.bashrc

RUN wget ftp://www.hensa.ac.uk/sites/distfiles.macports.org/gccmakedep/gccmakedep-1.0.2.tar.bz2
RUN tar jxf gccmakedep-1.0.2.tar.bz2 && cd ~/gccmakedep-1.0.2 && ./configure && make && sudo make install

# protobuf
RUN wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
RUN tar -xzvf protobuf-2.6.1.tar.gz
RUN cd protobuf-2.6.1 && ./configure && make && sudo make install

# zeromq
RUN wget http://download.zeromq.org/zeromq-4.0.5.tar.gz
RUN tar -xzvf zeromq-4.0.5.tar.gz
RUN cd zeromq-4.0.5 && ./configure && make && sudo make install

RUN mkdir -p /home/soccer/logs
RUN mkdir -p /home/soccer/.rcssserver/

VOLUME ["/home/soccer/logs"]

# TEMPORARY
RUN virtualenv --system-site-packages agent
RUN /home/soccer/agent/bin/pip install ipython numpy mock pdbpp protobuf Theano
RUN /home/soccer/agent/bin/pip install pyzmq

RUN sudo ldconfig

RUN mkdir -p /home/soccer/keepaway
ADD keepaway/ /home/soccer/keepaway/
RUN sudo chgrp -R soccer /home/soccer/keepaway/ && sudo chown -R soccer /home/soccer/keepaway/

ADD agent/ /home/soccer/agent/src/agent/
RUN /home/soccer/agent/bin/pip install -r /home/soccer/agent/src/agent/requirements.txt

WORKDIR /home/soccer/keepaway
RUN cd player && make depend && make
RUN cd tools && make

RUN mkdir -p "/home/soccer/~/.rcssserver"
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y supervisor
RUN sudo pip install supervisor-stdout

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ADD run_dql.sh /home/soccer/run_dql.sh
ADD run.sh /home/soccer/run.sh
ADD start.sh /home/soccer/start.sh

RUN sudo chgrp -R soccer /home/soccer/ && sudo chown -R soccer /home/soccer/

CMD ["/home/soccer/start.sh"]
