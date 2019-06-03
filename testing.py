import rev

# load a chart
chart = rev.Chart('examples/vega1.png', text_from=0)

##########################################################################################
# mark type classifier
# import rev.mark
# mark_clf = rev.mark.MarkClassifier()
# print mark_clf.classify([chart])

##########################################################################################
# feature extraction (single)
# import rev.text
# text_features = rev.text.feature_extractor.from_chart(chart)
# print text_features

##########################################################################################
# feature extraction (corpus)
# import rev.text
# text_features = rev.text.feature_extractor.from_chart(chart)
# print text_features

##########################################################################################
# text role classification
# import rev.text
# text_clf = rev.text.TextClassifier('default')
# text_type_preds = text_clf.classify(chart)
# print text_type_preds

##########################################################################################
# training text role classifier
import pandas as pd
import rev.text
data = pd.read_csv('data/features_all.csv')
features = data[rev.text.classifier.VALID_COLUMNS]
types = data['type']

text_clf = rev.text.TextClassifier()
text_clf.train(features, types)
# text_clf.save_model('models/text_role_classifier/text_type_classifier_new.pkl')

