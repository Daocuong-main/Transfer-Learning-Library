sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3-pip
sudo apt install python3-venv
sudo apt install python3.7
sudo apt install python3.7-distutils
sudo apt install python3.7-dev
python3.7 -m pip install virtualenv
python3.7 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip install -i https://test.pypi.org/simple/ tllib==0.4
