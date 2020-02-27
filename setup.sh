sudo apt update
sudo apt -y upgrade
sudo apt install -y python3-pip
sudo apt install -y python3-venv
python3.6 -m venv dl-scratch
cd planet-rep
source ./../dl-scratch/bin/activate
pip --version
pip3 install --upgrade pip==19.0.0
pip3 install -r requirements.txt
