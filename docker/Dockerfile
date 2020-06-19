# @file       Dockerfile
# @author     Igor Bogoslavskyi [igor.bogoslavskyi@gmail.com]
#
# Copyright (c) 2020  Ignacio Vizzo, all rights reserved
FROM gitlab.ipb.uni-bonn.de:4567/ipb-team/global/docker-images/ipb_opencv4
LABEL maintainer "ivizzo@uni-bonn.de"

CMD ["bash"]
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive

# Install fmt library
RUN git clone --depth 1 https://github.com/fmtlib/fmt.git -b 6.2.0 \
  && cd fmt \
  && mkdir build && cd build \
  && cmake .. && make -j all install

# Install google test
RUN git clone --depth 1 https://github.com/google/googletest.git  \
  && cd googletest \
  && mkdir build && cd build \
  && cmake .. && make -j all install

# Install Catch2
RUN git clone --depth 1 https://github.com/catchorg/Catch2.git \
  && cd Catch2 \
  && mkdir build && cd build \
  && cmake -DBUILD_TESTING=OFF .. && make -j all install

