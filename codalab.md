# How to run CoQA's DrQA + PGNet model on Codalab?

Before you run on Codalab, first you have to make sure you have a docker environment that can run your code. 
In the DrQA+PGNet case, we require torch=0.4.0, pycorenlp, torchtext==0.2.1, gensim.
You can create your own docker that satisifies your requirements, or you can use existing ones on https://hub.docker.com/.
We recommend https://hub.docker.com/r/floydhub which contains almost every deep-learning framework dockers.
We will use https://hub.docker.com/r/floydhub/pytorch/tags/, particularly, 0.4.0-gpu.cuda9cudnn7-py3.33.
However this docker does not contain pycorenlp, torchtext==0.2.1 gensim, so we install these requirements using pip after launching the environment.
Follow these steps one by one to run our model on the codalab.

## 1. Install codalab client to upload data from command line
See https://github.com/codalab/codalab-worksheets/wiki/CLI-Basics#installation

## 2. Create a worksheet
Go to https://worksheets.codalab.org and create a worksheet.
Say you call this username-coqa-baseline

## 3. Upload data to that worksheet
Run the following command from your terminal to switch to that worksheet first.

```
  cl work main::username-coqa-baseline
```

Downnload data to your local system and upload it to that worksheet. You can also use the web-interface to upload data if the data is in tar/zip format and then untar/unzip. If you use web-interface, you can skip steps 1 and 2.

```
  git clone --recurse-submodules git@github.com:stanfordnlp/coqa-baselines.git
  cl upload coqa-baselines
```

Add dev-file to your worksheet.

```
  cl add bundle 0xe25482 .
```

## 4. Install requirements and run the code
[run_on_codalab.sh](run_on_codalab.sh) installs the requirements and runs the code. On the codalab worksheet's web terminal, run the following command which specifies the docker, number of gpus, cpu memory, etc.

```cl run :coqa-dev-v1.0.json :coqa-baselines --request-docker-image floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.33 --request-network --request-gpus 1 --request-memory 6g 'sh run_on_codalab.sh' ```


