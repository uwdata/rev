"""
Script to predict type of text tole and update *-texts.csv file.

Usage:
    run_text_box_classifier.py train FEATURES_CSV OUTPUT_MODEL_PLK
    run_text_box_classifier.py single INPUT_PNG [--from_bbs=FROM] [--with_post]  [--pad=PAD]
    run_text_box_classifier.py multiple INPUT_LIST_TXT [--from_bbs=FROM] [--with_post]  [--pad=PAD]
    run_text_box_classifier.py (-h | --help)
    run_text_box_classifier.py --version

Options:
    --from_bbs FROM  1: from predicted1-bbs.csv  [default: 1]
                     2: from predicted2-bbs.csv
    --with_post      Boolean, run post processing?
    --pad PAD        Add padding to boxes [default: 0]
    -h --help        Show this screen.
    --version        Show version.

Examples:
  # train text role classifier
  python scripts/run_text_role_classifier.py train data/features_all.csv out.plk

  # run text role classifier in a chart to test
  python scripts/run_text_role_classifier.py single examples/vega1.png

  # run text role classifier in multiple charts
  python scripts/run_text_role_classifier.py multiple data/academic.txt
"""

from docopt import docopt
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

import pandas as pd

import rev.text
from rev import Chart, chart_dataset


def __classify(clf, chart, with_post=False, draw_debug=False, pad=0, save=False):
    print clf.classify(chart, with_post, draw_debug, pad, save)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    draw_debug = True

    if args['train']:
        features_file = args['FEATURES_CSV']
        output_file = args['OUTPUT_MODEL_PLK']

        data = pd.read_csv(features_file)
        features = data[rev.text.classifier.VALID_COLUMNS]
        types = data['type']

        text_clf = rev.text.TextClassifier()
        text_clf.train(features, types)
        text_clf.save_model(output_file)

    if args['single']:
        # clf = bc.load_classifier()
        image_name = args['INPUT_PNG']
        from_bbs = int(args['--from_bbs'])
        with_post = args['--with_post']
        pad = int(args['--pad'])
        print with_post

        chart = Chart(image_name, text_from=from_bbs)
        text_clf = rev.text.TextClassifier('default')
        pred_types = text_clf.classify(chart, with_post, draw_debug, pad, save=True)
        print pred_types

    if args['multiple']:
        chart_list = args['INPUT_LIST_TXT']
        from_bbs = int(args['--from_bbs'])
        with_post = args['--with_post']
        pad = int(args['--pad'])

        text_clf = rev.text.TextClassifier('default')
        # run in parallel
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=1, backend='multiprocessing')(
            delayed(__classify)(text_clf, chart, with_post, draw_debug, pad, True)
            for chart in chart_dataset(chart_list, from_bbs))

        # print 'Total boxes : %d' % sum(results)