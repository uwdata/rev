import copy
import pandas as pd
import sys
import os

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels

from .. import utils as u
from . import feature_extractor

# import warnings
# warnings.filterwarnings("ignore")

model_files = {
    'default': 'models/text_role_classifier/text_type_classifier.pkl',
    'testing': 'models/text_role_classifier/text_type_classifier_new.pkl'
}

# valid features
VALID_COLUMNS = [
    # 'fig_id',
    'vscore', 'hscore',
    'vrange', 'hrange',
    'vfreq', 'hfreq',
    # 'x',  'y',
    # 'x2', 'y2',
    # 'xc', 'yc',
    # 'w',  'h',
    # 'xp',  'yp',
    # 'x2p', 'y2p',
    'xcp', 'ycp',
    # 'wp',  'hp',
    'aspect',
    'ang',
    'quad',
    # 'bw', 'bh',
    'bwp', 'bhp',
    # 'u',  'v',
    # 'u2', 'v2',
    'uc', 'vc',
    # 'du', 'dv',
    # 'adu', 'adv',
    'rad',
    # 'text',
    # 'type'
]


class TextClassifier:
    def __init__(self, model_name=None):
        if model_name is None:
            # Pipeline: standardization -> svm
            my_svm = svm.SVC(C=100, gamma=0.1, class_weight='balanced', kernel='rbf')
            self._clf = make_pipeline(StandardScaler(), my_svm)
        else:
            model_file = model_files[model_name]
            self._clf = joblib.load(model_file)

    def train(self, features, types):
        """
        Train an svm model with the complete dataset.
        This classifier will be used for the following steps in the pipeline.
        :param features:
        :param types:
        :return:
        """
        print >> sys.stderr, 'fitting...',
        self._clf.fit(features, types)
        print 'DONE'

        print >> sys.stderr, 'evaluating...',
        pred_types = self._clf.predict(features)
        print 'DONE'

        cm = metrics.confusion_matrix(types, pred_types, labels=self._clf.classes_)
        u.print_cm(cm, labels=self._clf.classes_)
        print 'accuracy: ', metrics.accuracy_score(types, pred_types)
        print 'wrong boxes: ', sum(types != pred_types)

    def cross_validation(self, features, true_types, cv):
        labels = unique_labels(true_types)
        print 'total after sampling:', len(true_types)
        print pd.value_counts(true_types)[labels]

        # cross-validation
        pred_type = cross_val_predict(self._clf, features, true_types, cv=cv, n_jobs=-1)
        print metrics.classification_report(true_types, pred_type, target_names=labels)
        print 'Accuracy: ', metrics.accuracy_score(true_types, pred_type)

        cm = metrics.confusion_matrix(true_types, pred_type, labels=labels)
        u.print_cm(cm, labels=labels)

    def classify(self, chart, with_post=False, draw_debug=False, pad=0, save=False):
        """
        Classify text boxes in a chart and save them in a cvs file
        :param chart:
        :param with_post
        :param draw_debug
        :param save:  save pred_type in the *-texts.csv file
        :return:
        """
        if chart.text_from == 1 and not os.path.isfile(chart.text_boxes_filename):
            u.create_predicted1_bbs(chart)

        # extract boxes from chart
        fh, fw, _ = chart.image.shape
        text_boxes = copy.deepcopy(chart.text_boxes)
        for b in text_boxes:
            b.wrap_rect((fh, fw), padx=pad, pady=pad)

        pred_types = self.classify_from_boxes(text_boxes, (fh, fw), with_post)

        if save:
            for text_box, pred_type in zip(chart.text_boxes, pred_types):
                text_box._type = pred_type
            chart.save_text_boxes()

        return pred_types

    def classify_from_boxes(self, text_boxes, shape, with_post=False):
        """
        Classify text boxes
        :param text_boxes: bounding boxes
        :param shape: (fh, fw) figure height and width.
        :return:
        """
        data = feature_extractor.from_text_boxes(text_boxes, shape, 0, '')
        features = data[VALID_COLUMNS]

        # predict class
        pred_types = self._clf.predict(features)

        # if with_post:
        #     self.post_process(boxes)

        return pred_types

    def save_model(self, filename):
        joblib.dump(self._clf, filename)



