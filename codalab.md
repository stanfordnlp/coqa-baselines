# How to run a trained model on Codalab and submit it for evaluation?

We use [Codalab](https://worksheets.codalab.org/) to evaluate models and display their scores on the [leaderboard](https://stanfordnlp.github.io/coqa/).
We show you how to run DrQA+PGNet pretrained model but you could use a similar set up for your model.

Before you run on Codalab, first you have to make sure you have a docker environment that can run your code.
In the DrQA+PGNet case, we require torch=0.4.0, pycorenlp, torchtext==0.2.1, gensim.
You can create your own docker that satisifies your requirements, or you can use existing ones on https://hub.docker.com/.
We recommend https://hub.docker.com/r/floydhub which contains dockers for almost every deep-learning framework.
We will use https://hub.docker.com/r/floydhub/pytorch/tags/, particularly, 0.4.0-gpu.cuda9cudnn7-py3.33.
However this docker does not contain pycorenlp, torchtext==0.2.1 gensim, so we install them later using pip.
Follow these steps one by one to run our model on Codalab.

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

Download data to your local system and upload it to that worksheet. You can also use the web-interface to upload data if the data is in tar/zip format and then untar/unzip. If you use web-interface, you can skip steps 1 and 2.

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

```
  cl run :coqa-dev-v1.0.json :coqa-baselines 'sh coqa-baselines/run_on_codalab.sh' --request-docker-image floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.33 --request-network --request-gpus 1 --request-memory 6g
```

The resulting worksheet looks like this https://worksheets.codalab.org/worksheets/0xa8916802a3144c00a5cd6cd9f59768e4/

You can access the final predictions in [baseline_combined_model/predictions.combined.json](https://worksheets.codalab.org/rest/bundles/0x8bc73ba8cd904d778d7d1817b154fcaf/contents/blob/baseline_combined_model/)

Email sivar@stanford.edu when you can run your model successfully.

Please email these details:

1. link to your worksheet
2. cl run command
3. path to output predictions file
4. System name in this sample format:  ```BERT + MMFT + ADA (single model) Microsoft Research Asia```
