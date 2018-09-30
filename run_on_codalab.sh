echo "This file should be run in the parent directory of coqa-baselines"

echo "Installing requirements"
pip install pycorenlp torchtext==0.2.1 gensim
echo "Requirements Installed"

echo "Running corenlp server"
wget http://central.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.1/stanford-corenlp-3.9.1.jar
java -mx4g -cp stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 &
sleep 5
echo "Corenlp server running"

echo "Preprocessing the data"
python coqa-baselines/scripts/gen_pipeline_data.py --data_file coqa-dev-v1.0.json --output_file1 coqa.dev.combined.json --output_file2 seq2seq-dev-combined

echo "Downloading combined model"
wget https://nlp.stanford.edu/data/coqa/baseline_combined_model.tgz
tar -xvzf baseline_combined_model.tgz

echo "Running DrQA model"
python coqa-baselines/rc/main.py --testset coqa.dev.combined.json --n_history 2 --pretrained baseline_combined_model
python coqa-baselines/scripts/gen_pipeline_for_seq2seq.py --data_file coqa.dev.combined.json --output_file baseline_combined_model/combined-seq2seq-src.txt --pred_file baseline_combined_model/predictions.json

echo "Running PGNet model"
python coqa-baselines/seq2seq/translate.py -model baseline_combined_model/seq2seq_copy_acc_85.00_ppl_2.18_e16.pt -src baseline_combined_model/combined-seq2seq-src.txt -output baseline_combined_model/pred.txt -replace_unk -verbose -gpu 0
python coqa-baselines/scripts/gen_seq2seq_output.py --data_file coqa-dev-v1.0.json --pred_file baseline_combined_model/pred.txt --output_file baseline_combined_model/predictions.combined.json

echo "Evaluating the model"
python coqa-baselines/scripts/evaluate-v1.0.py --data-file coqa-dev-v1.0.json --pred-file baseline_combined_model/predictions.combined.json > eval.json
