# Reverse-Engineering Visualizations (REV)

REV([paper](http://idl.cs.washington.edu/papers/reverse-engineering-vis/)) is a text analysis pipeline which detects text elements in a chart, classifies their role (e.g., chart title, x-axis label, y-axis title, etc.), and recovers the text content using optical character recognition. It also uses a Convolutional Neural Network for mark type classification. Using the identified text elements and graphical mark type, it infers the encoding specification of an input chart image.

Our pipeline consist of the following steps:

* Text localization and recognition
* Text role classification 
* Mark type classification 
* Specification induction

## Installation
You first need to download our code:  
```sh
git clone git@github.com:uwdata/rev.git
```

Then, download the data and modes are in the following 
[link](https://drive.google.com/open?id=1Bg9hyxlt2szXj6CBWIIt3yInIjKEqPFx).
You have to unzip the files in the project folder. 


### Dependencies
* conda create -n rev python=2.7 opencv=3.4 pandas scikit-image scikit-learn 
docopt joblib
* caffe 1 (https://caffe.berkeleyvision.org/installation.html)

## Using our API
In this example we assume that we have the text elements from a chart. For a given image (`image.png`), text elements should be provided in a CSV file named `image-texts.csv` with the following format. 

```CSV
id,x,y,width,height,text,type
1,30,5,19,17,"45",y-axis-label
...
```
Check file `examples/vega1-texts.csv` for an example.

Text `type` can be: `title`, `x-axis-title`, `x-axis-label`, `y-axis-title`, 
`y-axis-label`, `legend-title`, `legend-label`, and `text-label`.

However, in most cases we do not have access to the text elements, then, we can infer them using our pipeline. Each step of our pipeline can be run independently.  



#### Text localization and recognition

#### Text role classification
```python
import rev.text

# feature extraction (single)
text_features = rev.text.feature_extractor.from_chart(chart)
print text_features

# feature extraction (corpus)
text_features = rev.text.feature_extractor.from_chart(chart)
print text_features

# text role classification
text_clf = rev.text.TextClassifier('default')
text_type_preds = text_clf.classify(chart)
print text_type_preds

# training text role classifier
import pandas as pd
data = pd.read_csv('data/features_all.csv')
features = data[rev.text.classifier.VALID_COLUMNS]
types = data['type']

text_clf = rev.text.TextClassifier()
text_clf.train(features, types)
text_clf.save_model('out.pkl')
```

#### Mark type classification
```python
import rev.mark
mark_clf = rev.mark.MarkClassifier()
print mark_clf.classify([chart])
```

#### Specification induction


## Scripts
Some usefull script to reproduce results from paper: 
```shell
# code to rate the text-role classifier (Table 4 from paper)
python scripts/rate_text_role_classifier.py features data/features_academic.csv
python scripts/rate_text_role_classifier.py features data/features_quartz.csv
python scripts/rate_text_role_classifier.py features data/features_vega.csv

# script to extract features
python scripts/run_feature_extraction.py multiple data/academic.txt out.csv

# train text-role classifier
python scripts/run_text_role_classifier.py train data/features_all.csv out.plk

# run text-role classifier in a chart to test
python scripts/run_text_role_classifier.py single examples/vega1.png

# run text-role classifier in multiple charts
python scripts/run_text_role_classifier.py multiple data/academic.txt
``` 
 
