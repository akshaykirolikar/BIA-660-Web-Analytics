#!/usr/bin/env python
# coding: utf-8

# In[597]:


import pandas as pd
import numpy as np
import re
import nltk
#nltk.download()

# In[599]:


def extract(text):
    result = re.findall(r'(\w+\s*\w*(?=,)),\s+(.*(?=\s+\()).*(\d{4})\)\:\s+\$(\d*,\d*)',text)
    return result


# In[600]:


import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en')
def tokenize(text,lemmatized = False,no_stopword = False):
    doc = nlp(text)
    stop_words = stopwords.words('english')
    if lemmatized :
        tokens = []
        for i in doc:
            tokens.append(i.lemma_)
        if no_stopword:
            tmp = tokens.copy()
            for i in tmp:
                #i = i.lower()
                if i in stop_words:
                    try:
                        tokens.remove(i)
                    except:
                        pass
            return tokens
        else:
            return tokens
    else:
        tokens = [i.text for i in doc]
        if no_stopword:
            tmp = tokens.copy()
            for i in tmp:
                if i in stop_words:
                        tokens.remove(i)
            return tokens
        else:
            return tokens


# In[601]:


from scipy.spatial import distance
from sklearn.preprocessing import normalize
def get_similarity(q1,q2,lemmatized = False,no_stopword = False):
    sim = None
    docs = q1+q2
    token_count = {}
    for i,j in enumerate(docs):
        tokens = tokenize(j,lemmatized=lemmatized,no_stopword=no_stopword)
        freq = nltk.FreqDist(tokens)
        token_count[i] = dict(freq)
    df = pd.DataFrame.from_dict(token_count,orient = 'index')
    df = df.fillna(0)
    tf = df.values
    doc_len = tf.sum(axis = 1)
    tf = np.divide(tf, doc_len[:,None])
    dfeq = np.where(tf>0,1,0)
    smoothed_idf = np.log(np.divide(len(docs)+1, np.sum(dfeq, axis=0)+1))+1
    smoothed_tf_idf = normalize(tf*smoothed_idf)
    sim=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))
    sim_list = []
    for i in range(0,500):
        sim_list.append(sim[i][i+500])
    return sim_list


# In[602]:


def predict(sim,ground_truth,threshold=0.5):
    predict = []
    count = np.sum(np.where(ground_truth>0,1,0),axis=0)
    count_same = 0
    for i in range(len(sim)):
        if sim[i]>threshold:
            predict.append(1.0)
        else:
            predict.append(0.0)
    predict = np.array(predict)
    for i in range(len(predict)):
        if predict[i] == ground_truth[i] == 1.0:
            count_same+=1
    recall = count_same/count
    return predict,recall    


# In[603]:


def evaluate(sim,ground_truth,threshold=0.5):
    predict_this,recall = predict(sim,ground_truth,threshold=threshold)
    correct_count = 0
    count = np.sum(np.where(predict_this>0,1,0))
    for i in range(len(predict_this)):
        if predict_this[i] == ground_truth[i] == 1.0:
            correct_count+=1
    precision = correct_count/count
    return precision,recall


# In[605]:


if __name__ == "__main__": 
    # Test Q1
    text='''Following is total compensation for other presidents at private colleges in Ohio in 2015:
            
            Grant Cornwell, College of Wooster (left in 2015): $911,651
            Marvin Krislov, Oberlin College (left in 2016):  $829,913
            Mark Roosevelt, Antioch College, (left in 2015): $507,672
            Laurie Joyner, Wittenberg University (left in 2015): $463,504
            Richard Giese, University of Mount Union (left in 2015): $453,800'''
    print("Test Q1")
    print(extract(text))
    
    data=pd.read_csv("quora_duplicate_question_500.csv",header=0)
    q1 = data["q1"].values.tolist()
    q2 = data["q2"].values.tolist()
    # Test Q2
    
    print("Test Q2")
    print("\nlemmatized: No, no_stopword: No")
    sim = get_similarity(q1,q2)
    pred, recall=predict(sim, data["is_duplicate"].values) 
    print(recall)
    
    print("\nlemmatized: Yes, no_stopword: No")
    sim = get_similarity(q1,q2, True)
    pred, recall=predict(sim, data["is_duplicate"].values) 
    print(recall)
    
    print("\nlemmatized: No, no_stopword: Yes")
    sim = get_similarity(q1,q2, False, True)
    pred, recall=predict(sim, data["is_duplicate"].values) 
    print(recall)
    print("\nlemmatized: Yes, no_stopword: Yes")
    sim = get_similarity(q1,q2, True, True)
    pred, recall=predict(sim, data["is_duplicate"].values) 
    print(recall)
    # Test Q3. Get similarity score, set threshold, and then
    print('\nTest Q3')
    sim = get_similarity(q1,q2)
    prec, rec = evaluate(sim, data["is_duplicate"].values, 0.5)
    print("\nlemmatized: No, no_stopword: No")
    print(prec,rec)
    
    sim = get_similarity(q1,q2,True)
    prec, rec = evaluate(sim, data["is_duplicate"].values, 0.5)
    print("\nlemmatized: Yes, no_stopword: No")
    print(prec,rec)
    
    sim = get_similarity(q1,q2,False,True)
    prec, rec = evaluate(sim, data["is_duplicate"].values, 0.5)
    print("\nlemmatized: No, no_stopword: Yes")
    print(prec,rec)
    
    sim = get_similarity(q1,q2,True,True)
    prec, rec = evaluate(sim, data["is_duplicate"].values, 0.5)
    print("\nlemmatized: Yes, no_stopword: Yes")
    print(prec,rec)



