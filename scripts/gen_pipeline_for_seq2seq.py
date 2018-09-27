"""
    Take a prediction file from DrQA and transform this to a seq2seq format.
"""

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--pred_file', '-p', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    args = parser.parse_args()

    with open(args.data_file) as f:
        dataset = json.load(f)

    questions = {}
    for i, datum in enumerate(dataset['data']):
        for qas in datum['qas']:
            idx = datum['id'] + ':' + str(qas['turn_id'])
            questions[idx] = ' '.join(qas['annotated_question']['word'])

    with open(args.pred_file) as f:
        predictions = json.load(f)

    f_out = open(args.output_file, 'w')
    for prediction in predictions:
        idx = prediction['id'] + ':' + str(prediction['turn_id'])
        f_out.write(questions[idx].lower() + ' || ' + prediction['answer'].replace('\n', ' ').lower() + '\n')
    f_out.close()
