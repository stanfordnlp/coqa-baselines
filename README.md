# coqa-baselines
We provide several baselines: conversational models, extractive reading comprehension models and their combined models for the [CoQA challenge](https://stanfordnlp.github.io/coqa/). See more details in the [paper](https://arxiv.org/abs/1808.07042).

## Requirements
1. PyTorch 0.4
2. pycorenlp
3. gensim

TODO

## Download
Download the dataset:
```bash
  mkdir data
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
```

Download pre-trained word vectors:
```bash
  mkdir wordvecs
  wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
  unzip -d wordvecs wordvecs/glove.42B.300d.zip
  wget -P wordvecs http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
  unzip -d wordvecs wordvecs/glove.840B.300d.zip
```

## Start a CoreNLP server

```bash
  mkdir lib
  wget -P lib http://central.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.1/stanford-corenlp-3.9.1.jar
  java -mx4g -cp lib/stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Conversational models
### Preprocessing
Generate the input files for seq2seq models --- needs to start a CoreNLP server:
```bash
  python scripts/gen_seq2seq_data.py --data_file data/coqa-train-v1.0.json --n_history 2 --lower --output_file data/seq2seq-train-h2
  python scripts/gen_seq2seq_data.py --data_file data/coqa-dev-v1.0.json --n_history 2 --lower --output_file data/seq2seq-dev-h2
```
Options:
* `n_history` can be changed to {0, 1, 2, ..} or -1.

Preprocess the data and embeddings:
```bash
  python seq2seq/preprocess.py -train_src data/seq2seq-train-h2-src.txt -train_tgt data/seq2seq-train-h2-tgt.txt -valid_src data/seq2seq-dev-h2-src.txt -valid_tgt data/seq2seq-dev-h2-tgt.txt -save_data data/seq2seq-h2 -lower -dynamic_dict -src_seq_length 10000
  PYTHONPATH=seq2seq python seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-h2.vocab.pt -output_file data/seq2seq.embed
```

### Training
Run a seq2seq (with attention) model:
```bash
   python seq2seq/train.py -data data/seq2seq-h2 -save_model models/seq2seq -word_vec_size 300 -pre_word_vecs_enc data/seq2seq.embed.enc.pt -pre_word_vecs_dec data/seq2seq.embed.dec.pt
```

Run a seq2seq+copy model:
```bash
   python seq2seq/train.py -data data/seq2seq-h2 -save_model models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/seq2seq.embed.enc.pt -pre_word_vecs_dec data/seq2seq.embed.dec.pt
```

## Reading comprehension models
### Preprocessing
Generate the input files for the reading comprehension (extractive question answering) model -- needs to start a CoreNLP server:
```bash
  python scripts/gen_drqa_data.py --data_file data/coqa-train-v1.0.json --output_file coqa.train.json
  python scripts/gen_drqa_data.py --data_file data/coqa-dev-v1.0.json --output_file coqa.dev.json
```

## Combined models

## Results

## Citation

```
  @article{reddy2018coqa,
     title={CoQA: A Conversational Question Answering Challenge},
     author={Reddy, Siva and Chen, Danqi and Manning, Christopher D},
     journal={arXiv preprint arXiv:1808.07042},
     year={2018}
   }
```

## License
MIT
