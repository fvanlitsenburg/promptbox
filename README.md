# Data analysis
- Document here the project: hackathon
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

To get the models locally:
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/fvanlitsenburg/promptbox.git

mkdir hf
cd hf

git clone https://huggingface.co/google/flan-t5-base
git clone https://huggingface.co/lmsys/fastchat-t5-3b-v1.0

git clone https://huggingface.co/tiiuae/falcon-7b


git clone https://huggingface.co/bigscience/bloom-560m
git clone https://huggingface.co/databricks/dolly-v2-3b

conda create -n python38 python=3.8
conda activate python38

git clone -b v1.17.1 https://github.com/deepset-ai/haystack.git

sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo mv /usr/local/bin/docker-compose /usr/bin/docker-compose

sudo systemctl start docker

cd promptbox
pip install -r requirements


#git clone git@hf.co:flan-t5-base
#git clone git@hf.co:fastchat-t5-3b-v1.0
'''

sudo apt install python3-pip

sudo apt update -y



# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for hackathon in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/hackathon`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "hackathon"
git remote add origin git@github.com:{group}/hackathon.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
hackathon-run
```

# Install

Go to `https://github.com/{group}/hackathon` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/hackathon.git
cd hackathon
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
hackathon-run
```
