# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:52:06 2020

@author: evefl
"""

from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora.dictionary import Dictionary
from gensim import models
from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

import numpy as np
import re
import pickle

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


""" Creates train and test sets from the corpus metadata.

Parameters
----------
None

Returns
-------
`X` (document vectors) and `y` (true class, literary or linguistic) 
for the train and test sets.
"""
def get_doc_vectors():
        
    path = ""
    metadata = get_metadata(path + "corpus.mm.metadata")   
    
    w2v_size = 100
    max_title_len = 35
    w2v_model = Word2Vec([word_tokenize(doc["title"])[:max_title_len] for doc in metadata], size=w2v_size)
    
    doc_X = np.zeros((len(metadata), w2v_size*max_title_len))
    doc_y = np.zeros(len(metadata))
    for index, doc in enumerate(metadata):
        
        # Get classification
        classification = -1
        # 0 = Linguistic, 1 = Literary; exclude other classes for now
        if 'Linguistic' in metadata[index]["category"] and 'Literary' not in metadata[index]["category"]:
            classification = 0
            
        if 'Linguistic' not in metadata[index]["category"] and 'Literary' in metadata[index]["category"]:
            classification = 1
        
        if classification >= 0:
            
            # Crop very long titles
            title = word_tokenize(metadata[index]["title"])[:max_title_len]
            title_vector = make_word_vector(w2v_model, title)
            
            doc_X[index] = title_vector
            doc_y[index] = classification

    X_train, X_test, y_train, y_test_gt = train_test_split(doc_X, doc_y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test_gt


"""Creates vector of concatenated word2vec embeddings of words in document

Parameters
----------
w2v_model : word2vec model
doc : list of str

Returns
-------
vector : np.array
    Vector representation of doc
"""
def make_word_vector(w2v_model, doc):
    
    vector = np.zeros(100*35)
    for w_index, word in enumerate(doc):
        if word in w2v_model:
            vector[w_index:w_index+100] = w2v_model[word]
            
    return vector
    
"""Runs a Naive Bayes or SVM classifier on the data.

Parameters
----------
classifier_type : {'svm', 'nb'}
    Whether to use an SVM or Naive Bayes classifier.

Returns
-------
y_pred : Predicted classes of test set data
"""  
def classify(classifier_type = "svm"):
    X_train, X_test, y_train, y_test_gt = get_doc_vectors()
    
    if classifier_type == "nb":
        classifier = GaussianNB()
    else:
        classifier = svm.SVC(C=1.5, tol=0.00001)
        
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    
    cm = metrics.confusion_matrix(y_test_gt, y_pred)
    print(cm, flush=True)
    score = classifier.score(X_test, y_test_gt)
    print("Accuracy:", score, flush=True)
    f1 = metrics.f1_score(y_test_gt, y_pred)
    print("F1:", f1)
    scores = metrics.precision_recall_fscore_support(y_test_gt, y_pred)
    print(scores, flush=True)
    report = metrics.classification_report(y_test_gt, y_pred, output_dict=True)
    
    return y_pred


"""Trains the neural network on the data.

Parameters
----------
all_x : Tensor
    Tensors of training data
y : Tensor
    Ground-truth tensors for training data
loss_fn : loss function

Returns
-------
model : torch.nn.module
    The trained model
"""
def train(all_x, y, loss_fn):
    
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    
    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False
        
    
    N, D_in, H, D_out = 1, all_x.size()[1], 100, 2
    
    print("N", N, "D_in", D_in, "H", H)
    n_classes = 2        
    
    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(H, n_classes))
    
    print("Begin training...", flush=True)
    
    learning_rate = 5e-4
    for t in range(800):
        
        total_loss = 0
        
        for x_index, x in enumerate(all_x[:100]):
                
            # Forward pass
            y_pred = model(x)
            #print(y_pred.size(), y.size())
            #print(x[:10], y[x_index], y_pred, flush=True)
            loss = loss_fn(y_pred, y[x_index])
            total_loss += loss
            
            # Zero the gradients before running the backward pass.
            model.zero_grad()
        
            # Backward pass
            loss.backward()
        
            # Update weights
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                    
        if t % 10 == 0:
            print("Epoch", t, "; total loss:", total_loss.item(), flush=True)
            torch.save(model, "prosody_classifier_2_epoch_" + str(t) + "_loss_" + str(loss.item()) + ".pt")
        
    return model

"""Tests the neural network on the data.

Parameters
----------
model : torch.nn.Module
all_x : Tensor
    Tensors of test data
y : Tensor
    Ground-truth tensors for test data
loss_fn : loss function
words : list, optional
    List of test examples corresponding to the items in all_x
    
Returns
-------
y_preds : Predicted classes for test data
"""
def test(model, all_x, y, loss_fn, words=None):
    with torch.no_grad():
        
        total_loss = 0
        y_preds = []
        for x_index, x in enumerate(all_x):
                
            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y[x_index])
            
            y_preds.append(y_pred)
            
            print("Predicted", y_pred, "Actual", y[x_index], flush=True)
        
            total_loss += loss
    
    # Evaluation
    print("Total loss:", total_loss)
    y_classes = [np.argmax(r) for r in y]
    yp_classes = [np.argmax(r) for r in y_preds]
    
    if words:
        for y_index, y in enumerate(y_classes):
            if not y == yp_classes[y_index]:
                print("Erred on: ", words[y_index])
    
    cm = metrics.confusion_matrix(y_classes, yp_classes)
    print(cm, flush=True)
    f1 = metrics.f1_score(y_classes, yp_classes)
    print("F1:", f1)
    scores = metrics.precision_recall_fscore_support(y_classes, yp_classes)
    print(scores, flush=True)
    
    true_zeros = [item for item in y_classes if item==0]
    true_ones = [item for item in y_classes if item==1]
    print(len(true_zeros), len(true_ones))
    
    pred_zeros = [item for item in yp_classes if item==0]
    pred_ones = [item for item in yp_classes if item==1]
    print(len(pred_zeros), len(pred_ones))
    
    # Get accuracy
    num_correct = 0
    for index, item in enumerate(yp_classes):
        if item==y_classes[item]:
            num_correct += 1
    
    accuracy = num_correct/len(y_classes)
    
    print("Accuracy:", accuracy)
    
    
    return y_preds


def get_metadata(fpath):
    
    text = open(fpath, 'r', encoding='utf8')
    print(text.readline())  # do not remove this line

    metadata = []
    ctr = 0
    for line in text:
        new_data = []
        items = re.split('(,)(?=(?:[^"]|"[^"]*")*$)', line)

        text = items[14]
        text = text.replace('"', ' ')
        new_data = {"id": items[0], "title": text, "category": items[26]}
        metadata.append(new_data)
    
    return metadata



"""Trains and tests the neural network.

Returns
-------
y_test_pred : Tensor
    Predicted classes on the test set.
""" 
def run_nn():
    loss_fn = torch.nn.MSELoss(reduction='sum')
    X_train, X_test, y_train, y_test_gt = get_doc_vectors()
    
    # Save Tensor-formatted ground truth
    y_train_tensor = []
    for cur_y in y_train:
        if cur_y==1:
            y_train_tensor.append([0, 1])
        else:
            y_train_tensor.append([1, 0])
    
    y_test_tensor = []
    for cur_y in y_test_gt:
        if cur_y==1:
            y_test_tensor.append([0, 1])
        else:
            y_test_tensor.append([1, 0])
        
    pickle.dump(X_train, open("prosody_x_train.p", "wb"))
    pickle.dump(X_test, open("prosody_x_test.p", "wb"))
    pickle.dump(y_train_tensor, open("prosody_y_train_tensors.p", "wb"))
    pickle.dump(y_test_tensor, open("prosody_y_test_tensors.p", "wb"))
          
    
    model = train(torch.FloatTensor(X_train), torch.FloatTensor(y_train_tensor), loss_fn)
    torch.save(model, "prosody_classifier.pt")
    #model = torch.load("prosody_classifier.pt")
    X_test = pickle.load(open("prosody_x_test.p", "rb"))
    y_test_tensor = pickle.load(open("prosody_y_test_tensors.p", "rb"))
    y_test_pred = test(model, torch.FloatTensor(X_test), torch.FloatTensor(y_test_tensor), loss_fn)
    
    return y_test_pred

"""Creates a TF-IDF model from corpus.mm
"""
def corpus_tfidf():
    path = "" 
    corpus = MmCorpus(path + "corpus.mm")
    id2word = Dictionary.load(path + 'corpus.mm.dict')
    
    # TF-IDF the corpus
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    tfidf.save("5_topics_tfidf_only.model")
    
    lda_model_tfidf = models.LdaModel(corpus_tfidf, num_topics=5, id2word=id2word)#models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=id2word, passes=2, workers=4) # better model
    print('\nPerplexity: ', lda_model_tfidf.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
        
    lda_model_tfidf.save(path + "5_topics_test.model")
    lda_model_tfidf.wv.save(path + "5_topics_test_kv.model")

classify()
#run_nn()


    
    