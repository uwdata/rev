import numpy as np
import caffe

models = {
    'revision': {
        'path': 'models/mark_classifier/revision/',
        'model_file': 'deploy.prototxt',
        'weights_file': 'model_iter_50000.caffemodel',
        'mean_file': 'ilsvrc_2012_mean.npy',
        'categories_file': 'categories.txt'
    },
    'charts5cats': {
        'path': 'models/mark_classifier/charts5cats/',
        'model_file': 'deploy.prototxt',
        'weights_file': 'model_iter_50000.caffemodel',
        'mean_file': 'ilsvrc_2012_mean.npy',
        'categories_file': 'categories.txt'
    }
}


class MarkClassifier:
    def __init__(self, model_name=None):
        model = models[model_name if model_name is not None else 'charts5cats']
        print model['path']+model['model_file']
        print model['path']+'snapshots/'+model['weights_file']
        print model['path']+model['mean_file']
        self._net = caffe.Classifier(
            model_file=model['path']+model['model_file'],
            pretrained_file=model['path']+'snapshots/'+model['weights_file'],
            mean=np.load(model['path']+model['mean_file']).mean(1).mean(1),
            channel_swap=(2, 1, 0),
            raw_scale=255)

        self._categories = np.genfromtxt(model['path']+model['categories_file'], dtype=None, encoding=None)

    @property
    def categories(self):
        return self._categories

    def train(self):
        pass

    def classify(self, charts):
        def chunks(l, n):
            # Yield successive n-sized chunks from l.
            for i in xrange(0, len(l), n):
                yield l[i:i + n]

        all_predictions = []
        for block_charts in chunks(charts, 100):
            inputs = [caffe.io.load_image(chart.filename) for chart in block_charts]

            predictions = self._net.predict(inputs, True)
            predictions = predictions.argmax(1)
            all_predictions.append(predictions)

        predictions = np.hstack(all_predictions)

        return self._categories[predictions]
