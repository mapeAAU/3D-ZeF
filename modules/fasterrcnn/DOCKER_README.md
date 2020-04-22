### Setup Docker with CUDA support

* Download and install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

* Install Docker (x86_64 Ubuntu 16.04/18.04 - see https://docs.docker.com/install/linux/docker-ce/ubuntu for more information):

        $ sudo apt-get update
        $ sudo apt-get install \
                apt-transport-https \
                ca-certificates \
                curl \
                gnupg-agent \
                software-properties-common 

        $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        $ sudo apt-get update
        $ sudo apt-get install docker-ce docker-ce-cli containerd.io
    
* Install Nvidia Docker (x86_64 Ubuntu 16.04/18.04 - see https://github.com/NVIDIA/nvidia-docker for more information)

        $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

        $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        $ sudo systemctl restart docker
   
### Clone this repository
        $ git clone git@bitbucket.org:aauvap/pytorch_docker.git

### Build docker image
        $ cd pytorch_docker/docker/pytorch/
        $ sudo docker build -t pytorch:vap .
        
### Adjust the ```pytorch.sh``` file
* Change 'user' to your Ubuntu user-name
* Etc.

### Check that CUDA is avilable
* Go to root folder

        $ sudo ./pytorch.sh
        $ nvidia-smi
        
* `nvidia-smi` should return a description of the installed nvidia graphics driver

        $ python -c 'import torch; print(torch.cuda.is_available())'
        
* The output should be `True`
        
# DEBUG
* If you get the error ```setfacl: Option -m: Invalid argument near character 5``` it is because the username in ```pytorch.sh``` is invalid and not corresponds to your Ubuntu username.
