# Promptbox

Promptbox is a "private GPT" solution that allows you to perform analysis on your own documents within a secure, circumscribed environment.

More information can be found in this Medium article:
https://medium.com/@fvanlitsenburg/building-a-privategpt-with-haystack-part-1-why-and-how-de6fa43e18b

# Installation

## Downloading Large Language Models from Huggingface

Promptbox runs on locally stored models. It has been built to run on Ubuntu 20.04.

To get the models locally, we need git-lfs:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

In the file `ui/utils.py`, we reference the model paths. Therefore, we need to create this file at the same directory level as where we'll install Promptbox.

```
mkdir hf
cd hf
```

Promptbox uses flan-t5-base and fastchat-t5-3b-v1.0. Other models can be used as well. To use other models, specify them in line 15 of `ui/Home.py`.

```
git clone https://huggingface.co/google/flan-t5-base
git clone https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
```

## Downloading Promptbox and Haystack repositories

git clone -b v1.17.1 https://github.com/deepset-ai/haystack.git
git clone https://github.com/fvanlitsenburg/promptbox.git

## Running Haystack and Promptbox

Assuming you have created a separate python environment in pyenv or conda.

Assuming you have docker-compose installed and running.

First,

```
cd haystack
docker-compose up
```

In a separate terminal, let's get Promptbox running.

```
cd promptbox
pip install -r requirements
streamlit run ui/Home.py
```

That's it! Promptbox should now be running on localhost:8501.
