FROM      ubuntu:14.04
MAINTAINER Mateusz Kurek master.mateusz@gmail.com


RUN apt-get update
RUN apt-get install -y gcc python checkinstall flex build-essential libboost-all-dev vim wget pkg-config qt4-dev-tools libqt4-dev libqt4-core libqt4-gui libqt4-gui libqt4-gui libqt4-network libqt4-opengl libqt4-dev libaudio-dev libxt-dev libpng12-dev libxi-dev libxrender-dev libfreetype6-dev libfontconfig1-dev libglib2.0-dev git

RUN mkdir -p /home/soccer
RUN groupadd -r soccer && useradd -r -g soccer --shell /bin/bash --home /home/soccer soccer
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

WORKDIR /home/soccer/keepaway
RUN cd player && make
RUN cd tools && make

VOLUME  ["/home/soccer/keepaway"]
