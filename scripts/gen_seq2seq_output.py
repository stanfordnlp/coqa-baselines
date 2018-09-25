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

    output = []
    for i, datum in enumerate(dataset['data']):
        print('processing {}/{}...'.format(i, len(dataset['data'])))
        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            output.append({'id': datum['id'], 'turn_id': question['turn_id']})

    predictions = []
    with open(args.pred_file) as f:
        for line in f.readlines():
            predictions.append(line.strip())

    assert len(output) == len(predictions)
    for out, pred in zip(output, predictions):
        out['answer'] = pred

    with open(args.output_file, 'w') as outfile:
        json.dump(output, outfile, indent=4)
