
# coding: utf-8

# In[1]:


## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 


# In[116]:


import os
from collections import Counter
from queue import Queue
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
# import tensorflow as tf
# import keras
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
import xgboost as xgb

import util


# In[117]:


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in range(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].items():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    


# In[118]:


## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

def sys_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping el.tag to the number of times each system call 
      is made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c[el.tag] += 1
    return c

def n_gram_sys_call_count_feats(tree):
    c = Counter()
    n = 3
    in_all_section = False
    q = Queue(maxsize = n - 1)
    
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section and not q.full():
            q.put(el.tag)
        elif in_all_section:
            key = "-".join([str(elt) for elt in list(q.queue)]) + "-" + el.tag
            c[key] += 1
            q.get()
            q.put(el.tag)
    return c

def naive_bayes(X_train, t_train, X_test, global_feat_dict, test_features_dict, lamb_param):
    sums = np.zeros(len(util.malware_classes))

    for i in range(X_train.shape[0]):
        sums[t_train[i]] += np.sum(X_train[i])
    
    prior = [3.69,1.62,1.2,1.03,1.33,1.26,1.72,1.33,52.14,.68,17.56,1.04,12.18,1.91,1.3]
    
    scores = np.zeros((X_test.shape[0], len(util.malware_classes)))
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            for k in range(len(util.malware_classes)):
                scores[i][k] += np.log((X_test[(i,j)] + lamb_param) / (sums[k] + len(test_features_dict) * lamb_param))

    scores = scores.T
    for k in range(len(scores)):
        scores[k] += np.log(prior[k] / 100) * 110
    scores = scores.T

    preds = np.argmax(scores, axis = 1)
    print(preds[:30])
    print(scores[:30])
    print(preds[600:])

    return preds

def acc(preds, t_validate):
    return np.sum(preds == np.array(t_validate)) / len(preds)


# In[122]:


# Metadata feature extraction
def metadata_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping attributes startreason, terminationreason, executionstatus
      to reason string
    """
    c = Counter()
    for el in tree.iter():
        # ignore everything outside the "process" element
        if el.tag == "process":
            for key in el.attrib:
                if key == "startreason" or key == "terminationreason" or key == "executionstatus":
                    c[el.attrib[key]] += 1
    return c

def time_elapsed(tree):
    c = Counter()
    for el in tree.iter():
        # ignore everything outside the "process" element
        if el.tag == "process":
            for key, val in el.attrib.items():
                if key == "starttime":
                    (m, sms) = val.split(':')
                    (s, ms) = sms.split('.')
                    start_sec = int(m) * 60 + int(s) + int(ms) * 0.001
                elif key == "terminationtime":
                    (m, sms) = val.split(':')
                    (s, ms) = sms.split('.')
                    end_sec = int(m) * 60 + int(s) + int(ms) * 0.001
                    c["total_time"] += (end_sec - start_sec)
    return c

def file_size(tree):
    c = Counter()
    for el in tree.iter():
        # ignore everything outside the "process" element
        if el.tag == "process":
            for key, val in el.attrib.items():
                if key == "filesize":
                    sz = int(val)
                    c["file_size"] += sz
    return c


# In[125]:


# Function to test different metadata features; returns a vector of that feature's values
def test_feat(feat_func, attrib_name):
    direc = "train"
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    tests = []
    for datafile in os.listdir(direc):
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        classes.append(util.malware_classes.index(clazz))
        tree = ET.parse(os.path.join(direc,datafile))
        tests.append(feat_func(tree)[attrib_name])
#     print (tests)
    return tests, classes

times, tm_classes = test_feat(time_elapsed, "total_time")
file_sizes, sz_classes = test_feat(file_size, "file_size")


# In[143]:


import matplotlib.pyplot as plt

# Shows time elapsed as a feature
plt.figure()
plt.scatter(times, tm_classes, color="black")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Malware index")
plt.title("Elapsed time for various malware types")
plt.show()

times_array = np.vstack((tm_classes, times)).T
for i in range(15):
    print ("Mw idx {}:".format(i), np.mean(times_array[times_array[:,0] == i]))

# Shows filesize as a feature
plt.figure()
plt.scatter(file_sizes, sz_classes, color="black")
plt.xlabel("Size of file")
plt.ylabel("Malware index")
plt.title("Filesize for various malware types")
plt.show()

fsz_array = np.vstack((sz_classes, file_sizes)).T
for i in range(15):
    print ("Mw idx {}:".format(i), np.mean(fsz_array[fsz_array[:,0] == i]))


# In[144]:


## The following functions do the feature extraction, learning, and prediction

train_dir = "train"
test_dir = "test"
outputfile = "sample_predictions.csv"  # feel free to change this or take it as an argument

# TODO put the names of the feature functions you've defined above in this list
ffs = [n_gram_sys_call_count_feats, metadata_feats, time_elapsed, file_size]

# extract features
print ("extracting training features...")
X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
print ("done extracting training features")
print ()


# In[163]:


split = int(np.floor(X_train.shape[0] * 0.8))
s = np.arange(X_train.shape[0])
np.random.shuffle(s)

X_train_samp = X_train[s[:split]]
t_train_samp = t_train[s[:split]]
X_val_samp = X_train[s[split:]]
t_val_samp = t_train[s[split:]]


# In[165]:


# TODO train here, and learn your classification parameters
print ("learning...")
# learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
clf = RandomForestClassifier()
clf2 = AdaBoostClassifier(RandomForestClassifier())
# clf.fit(X_train[s[:split]], t_train[s[:split]])
# clf2.fit(X_train[s[:split]], t_train[s[:split]])

# if want to actually submit these results
clf.fit(X_train, t_train)
clf2.fit(X_train, t_train)


print ("done learning")
print ()


# In[181]:


# training XGBoost
xgb_params = {'booster': 'dart', 'learning_rate': 0.1, 'n_estimators': 350, 'min child weight': 5, 'max_depth': 3}
xgb_clf = xgb.XGBClassifier(**xgb_params)
# xgb_clf.fit(X_train_samp, t_train_samp)
xgb_clf.fit(X_train, t_train)  # if want to actually submit these results

xgb_pred = xgb_clf.predict(X_val_samp)


# In[184]:


preds = clf.predict(X_train[s[split:]])
print (acc(preds, t_train[s[split:]]))
preds = clf2.predict(X_train[s[split:]])
print (acc(preds, t_train[s[split:]]))

# xgb_pred = np.round(xgb_pred).astype(int)
# print (xgb_pred)
# print (preds)
print (acc(xgb_pred, t_val_samp))


# In[167]:


# get rid of training data and load test data
# del X_train
# del t_train
# del train_ids
print ("extracting test features...")
X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
print ("done extracting test features")
print ()
# preds_bayes = naive_bayes(X_train, t_train, X_test, global_feat_dict, test_features_dict, 1)
# split = 2300
# preds_bayes = naive_bayes(X_train[:split], t_train[:split], X_train[split:], global_feat_dict, global_feat_dict, 5)


# In[182]:


# TODO make predictions on text data and write them out
print ("making predictions...")
# preds = np.argmax(X_train.dot(learned_W),axis=1)

# preds = clf2.predict(X_test)
preds = xgb_clf.predict(X_test)

print (preds)
print (acc(preds, t_train))
print ("done making predictions")
print ()

print ("writing predictions...")
util.write_predictions(preds, test_ids, outputfile)
print ("done!")


# In[183]:


# reorders predictions file accordingly
get_ipython().magic('run reorder_submission.py "sample_predictions.csv" "sample_predictions.csv"')

