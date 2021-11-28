# Environment Setup

## Python

``` sh
# Install Python 3.7
sudo apt-get install build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
sudo apt-get install python3-pip python3.7-dev
sudo apt-get install python3.7

# Handle the System Configuration
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# Select Python Version
sudo update-alternatives --config python3

```

## Numba

Install Numba for aarch64 using the guide [here](https://github.com/jefflgaol/Install-Packages-Jetson-ARM-Family)


```sh
pip3 install Cython
pip3 install numpy
pip3 install llvmlite
```

```
sudo apt-get install llvm-config-4.0
sudo su -
export LLVM_CONFIG=$(which llvm-config-4.0)
pip install numba
```

## Open3D


