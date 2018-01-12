FROM ubuntu:14.04

RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  libssl-dev \
  libzmq3-dev

RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
  git checkout 5beb83c46e91abd273c192a3fa782b62217072a6

RUN cd /root/torch && bash install-deps && ./install.sh

WORKDIR /root/

ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

RUN luarocks install tds
RUN git clone https://github.com/OpenNMT/OpenNMT /root/opennmt && cd /root/opennmt && \
  git checkout 5e55e9dfa0eabee2a64c6415171d4d1da2304233

EXPOSE 5556

RUN apt-get install -y libzmq-dev && luarocks install dkjson && \
  luarocks install lua-zmq ZEROMQ_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZEROMQ_INCDIR=/usr/include

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache
# CMD th tools/translation_server.lua -host 0.0.0.0 -port 5556 -model ~/data/CPU_best_seq2seq-featured-model_epoch7_84.85.t7 -beam_size 12

RUN luarocks install bit32
