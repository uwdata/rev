"""
Script to compute accuracy.

Usage:
  rate_mark_type_classifier.py MODEL_NAME TEST_FILE [--show]
  rate_mark_type_classifier.py (-h | --help)
  rate_mark_type_classifier.py --version

Options:
  --show              Show incorrect predictions.
  -h --help           Show this screen.
  --version           Show version.

Example:
  python scripts/rate_mark_type_classifier.py revision models/mark_classifier/revision/test.txt
"""
from docopt import docopt
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import rev.mark
import rev.utils as u


def report_accuracy(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels)

    print 'accuracy: ', accuracy_score(y_true, y_pred)
    u.print_cm(cm, labels=labels)
    print classification_report(y_true, y_pred, target_names=labels)


def main():
    model_name = args['MODEL_NAME']
    test_file = args['TEST_FILE']

    # loading model
    mark_clf = rev.mark.MarkClassifier(model_name)

    # loading testing data
    test_data = np.genfromtxt(test_file, dtype=None)
    # test_data = test_data[0:100]
    test_charts = [rev.Chart(item[0]) for item in test_data]
    true_types = [mark_clf.categories[item[1]] for item in test_data]

    # classifying and evaluating
    pred_types = mark_clf.classify(test_charts)
    report_accuracy(true_types, pred_types, mark_clf.categories)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if args['--show']:
        SHOW = True

    main()
