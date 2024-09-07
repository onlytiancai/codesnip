https://pytorch.org/cppdocs/installing.html

# 下载解压 torch c++ 版本

    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    unzip libtorch-shared-with-deps-latest.zip

# 升级 cmake

    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y cmake

# 升级 gcc, g++

    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install gcc-9 g++-9
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# 编译 torch c++ 项目

    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/home/action/download/libtorch ..
    cmake --build . --config Release
    ./torch-test
