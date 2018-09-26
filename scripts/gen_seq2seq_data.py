"""
    This file takes a CoQA data file as input and generates the input files for the conversational models.
"""

import argparse
import json
import time
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def tokenize_text(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json'})
    tokens = []
    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            tokens.append(_str(token['word']))
    return ' '.join(tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--n_history', type=int, default=0,
                        help='leverage the previous n_history rounds of Q/A pairs'
                             'if n_history == -1, use all history')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--output_file', '-o', type=str, required=True)
    args = parser.parse_args()

    f_src = open('{}-src.txt'.format(args.output_file), 'w')
    f_tgt = open('{}-tgt.txt'.format(args.output_file), 'w')

    with open(args.data_file) as f:
        dataset = json.load(f)

    start_time = time.time()
    data = []
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        context_str = tokenize_text(datum['story'])
        assert len(datum['questions']) == len(datum['answers'])

        history = []
        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            question_str = tokenize_text(question['input_text'])
            answer_str = tokenize_text(answer['input_text'])

            full_str = context_str + ' ||'
            if args.n_history < 0:
                for i, (q, a) in enumerate(history):
                    d = len(history) - i
                    full_str += ' <Q{}> '.format(d) + q + ' <A{}> '.format(d) + a
            elif args.n_history > 0:
                context_len = min(args.n_history, len(history))
                for i, (q, a) in enumerate(history[-context_len:]):
                    d = context_len - i
                    full_str += ' <Q{}> '.format(d) + q + ' <A{}> '.format(d) + a
            full_str += ' <Q> ' + question_str
            if args.lower:
                full_str = full_str.lower()
                answer_str = answer_str.lower()
            f_src.write(full_str + '\n')
            f_tgt.write(answer_str + '\n')
            history.append((question_str, answer_str))

    f_src.close()
    f_tgt.close()
