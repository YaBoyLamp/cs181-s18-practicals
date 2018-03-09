import os
from collections import Counter
from Queue import Queue
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util
from collections import namedtuple
import gensim

def d2v_features (direc, ng):

    def ngrams(input_list, n):
        return [' '.join(elt) for elt in zip(*[input_list[i:] for i in range(n)])]

    features = {}
    classes = {}
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        # add target class if this is training data
        try:
            classes[id_str] = util.malware_classes.index(clazz)
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes[id_str] = -1
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        in_all_section = False
        for el in tree.iter():
            # ignore everything outside the "all_section" element
            if el.tag == "all_section" and not in_all_section:
                in_all_section = True
            elif el.tag == "all_section" and in_all_section:
                in_all_section = False
            elif in_all_section:
                if id_str not in features:
                    features[id_str] = []
                features[id_str].append(el.tag)

    for key in features:
        features[key] = ngrams(features[key], ng)

    return features, classes

def train_d2v(features, classes):
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for key in features:
        docs.append(analyzedDocument(features[key], [key]))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=100, window_size=15, sampling_threshold = 1e-5, negative_size = 5)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    # model=gensim.models.doc2vec.Doc2Vec(docs, size=200, window=300, min_count = 1, workers=4, train_epoch=100)

    return model

def test_d2v (model, features, classes):

    def most_common(lst):
        data = Counter(lst)
        return max(lst, key=data.get)
    pred = {}
    for key in features:
        inferred_vector = model.infer_vector(features[key])
        neighbors = [item[0] for item in model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))]
        pred[key] = classes[most_common(neighbors)]
        # pred[key] = classes[model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))[0][0]]
    return pred

def d2v_acc (pred, classes):
    successes = 0
    total = 0
    for key in pred:
        if pred[key] == classes[key]:
            successes += 1
        total += 1

    return float(successes) / total

def main():

    features, classes = d2v_features('train', 1)
    model = train_d2v(features, classes)
    features, _ = d2v_features('test', 1)
    pred = test_d2v(model, features, classes)

    with open('d2v_predictions.csv', 'a') as f:
        f.write('ID,Prediction\n')
        for key in pred:
            f.write(key + ',' + str(pred[key]) + '\n')

if __name__ == "__main__":
    main()