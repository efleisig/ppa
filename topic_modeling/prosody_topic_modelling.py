# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:21:00 2019

@author: evefl
"""

from pprint import pprint
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora.dictionary import Dictionary
from gensim import models


# set path to wherever you download the files
path = '/data/'

corpus = MmCorpus("corpus.mm")#MmCorpus('%scorpus.mm' % path) # BOW
id2word = Dictionary.load('corpus.mm.dict')#'%scorpus.mm.dict' % path)



for doc in corpus[:1]:
    for word in doc[:2000]:
        print(word)
        print(id2word[word[0]])

# TF-IDF the corpus

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#
#for doc in corpus_tfidf: # preview tfidf scores for first document
#    pprint(doc)
#    break
#
#lda_model = models.LdaMulticore(corpus, num_topics=10, id2word=id2word, passes=2, workers=2)
#
#for idx, topic in lda_model.print_topics(-1):
#    print('Topic: {} \nWords: {}'.format(idx, topic)) # plain LDA w/o TF-IDF
#    
lda_model_tfidf = models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=id2word, passes=2, workers=4) # better model
print('\nPerplexity: ', lda_model_tfidf.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


#print('\nPerplexity: ', lda_model_tfidf.log_perplexity(corpus))