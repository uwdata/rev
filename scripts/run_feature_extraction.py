"""
Script to run text box feature extraction

Usage:
  run_feature_extraction.py single INPUT_PNG OUTPUT_CSV [--from_bbs=FROM]
  run_feature_extraction.py multiple INPUT_LIST_TXT OUTPUT_CSV [--from_bbs FROM]
  run_feature_extraction.py (-h | --help)
  run_feature_extraction.py --version

Options:
  --from_bbs FROM   0: from bbs.csv             [default: 0]
                    1: from pred1-bbs.csv
                    2: from pred2-bbs.csv
   -h --help                Show this screen.
  --version               Show version.

Examples:
  python scripts/run_feature_extraction.py single examples/vega1.png out.csv
  python scripts/run_feature_extraction.py multiple data/academic.txt out.csv
"""
from docopt import docopt
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

from rev import Chart, chart_dataset
from rev.text.feature_extractor import from_chart


def sample_group(data, samples_per_group):
    def sampling(group, num_samples):
        if num_samples < 0 or num_samples > len(group):
            num_samples = len(group)
        return group.sample(num_samples)

    return data.groupby('type').apply(sampling, num_samples=samples_per_group)


def main():
    from_bbs = int(args['--from_bbs'])
    if args['single']:
        image_name = args['INPUT_PNG']

        chart = Chart(image_name, text_from=from_bbs)
        text_features = from_chart(chart)
        text_features.to_csv(args['OUTPUT_CSV'], index=False)

    if args['multiple']:
        chart_list = args['INPUT_LIST_TXT']

        # run in parallel
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=1, backend='multiprocessing')(
            delayed(from_chart)(chart) for chart in chart_dataset(chart_list, from_bbs))

        df = pd.concat(results)
        df.to_csv(args['OUTPUT_CSV'], index=False)


if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    main()
