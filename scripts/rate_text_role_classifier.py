"""
Script to compute accuracy.

Usage:
  rate_text_role_classifier.py features FEATURES_CSV [--group_size GROUP_SIZE]
  rate_text_role_classifier.py (-h | --help)
  rate_text_role_classifier.py --version

Options:
 --group_size GROUP_SIZE  Number of elements per group. -1 means not sampling [Default: -1].
 -h --help                Show this screen.
  --version               Show version.

Examples:
  python scripts/rate_text_role_classifier.py features data/features_academic.csv
  python scripts/rate_text_role_classifier.py features data/features_quarts.csv
  python scripts/rate_text_role_classifier.py features data/features_vega.csv
"""
from docopt import docopt
import pandas as pd
import numpy as np

import rev.text


def sample_group(data, samples_per_group):
    def sampling(group, num_samples):
        if num_samples < 0 or num_samples > len(group):
            num_samples = len(group)
        return group.sample(num_samples)

    return data.groupby('type').apply(sampling, num_samples=samples_per_group)


def main():
    if args['features']:
        features_file = args['FEATURES_CSV']
        samples_per_group = int(args['--group_size'])

        # loading model
        text_clf = rev.text.TextClassifier()

        # loading test data
        np.random.seed(seed=0)
        data = pd.read_csv(features_file)
        data = sample_group(data, samples_per_group)
        test_features = data[rev.text.classifier.VALID_COLUMNS]
        true_types = data['type']

        # cross-validation
        text_clf.cross_validation(test_features, true_types, cv=5)

        return


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    main()
