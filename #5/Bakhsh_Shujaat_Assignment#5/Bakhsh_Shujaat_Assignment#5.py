
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from rank_bm25 import BM25Okapi

def classify(training_file,testing_file):
    data = pd.read_csv(training_file)
    
    print("\n")
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', MultinomialNB())
                       ])
    parameters = {'tfidf__min_df':[1, 2,3],
                  'tfidf__stop_words':[None,"english"],
                  'clf__alpha': [0.5,1.0,2.0],
    }
    metric =  "f1_macro"
    gs_clf = GridSearchCV(text_clf, param_grid=parameters,scoring=metric, cv=5)
    gs_clf = gs_clf.fit(data["text"], data["label"])
    print("Training metrics")
    for param_name in gs_clf.best_params_:
        print(param_name,": ",gs_clf.best_params_[param_name])

    print("best f1 score:", gs_clf.best_score_)
    print("\n")
    tfidf_vect = TfidfVectorizer(min_df=1).fit(data['text'])
    dtm= tfidf_vect.transform(data["text"])

    clf = MultinomialNB(alpha=2).fit(dtm,data['label'])
    test_data = pd.read_csv(testing_file)
    
    dtm_test = tfidf_vect.transform(test_data['text'])
    predicted=clf.predict(dtm_test)
    print("\n")
    print("Prediction Sample")
    print(predicted[0:3])
    print(test_data.label[0:3])
    
    print("\n")
    print(classification_report(test_data.label, predicted))
    
    predict_p=clf.predict_proba(dtm_test)

    binary_y = np.where(test_data['label']==2,1,0)
    y_pred = predict_p[:,1]
    
    fpr, tpr, thresholds = roc_curve(binary_y, y_pred,pos_label=1)
    print(auc(fpr, tpr))
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC of Naive Bayes Model')
    plt.show()
    precision, recall, thresholds = precision_recall_curve(binary_y,y_pred, pos_label=1)

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision_Recall_Curve of Naive Bayes Model')
    plt.show()



def impact_of_sample_size(train_file):
    data = pd.read_csv(train_file)
    data.head()
    step = 800
    sample = []
    average_f1_NB = []
    average_f1_SVM = []
    average_auc_NB = []
    average_auc_SVM = []
    while step<=len(data):
        data_sample = data[: step]
        tfidf_vect = TfidfVectorizer(stop_words='english')
        dtm_sample = tfidf_vect.fit_transform(data_sample['text'])
        metrics = ["f1_macro","roc_auc"]

        clf_NB = MultinomialNB()
        clf_SVM =svm.LinearSVC()

        binary_y = np.where(data_sample.label==2,1,0)

        cv_NB = cross_validate(clf_NB, dtm_sample, binary_y, scoring=metrics, cv=5, return_train_score=True)

        sample.append(step)

        average_f1_NB.append(cv_NB['test_f1_macro'].mean())
        average_auc_NB.append(cv_NB['test_roc_auc'].mean())

        cv_SVM = cross_validate(clf_SVM, dtm_sample, binary_y,scoring=metrics, cv=5,return_train_score=True)

        average_f1_SVM.append(cv_SVM['test_f1_macro'].mean())
        average_auc_SVM.append(cv_SVM['test_roc_auc'].mean())
        step+=400
    plt.figure()
    plt.plot(sample, average_f1_NB, color='orange', lw=2)
    plt.plot(sample,average_f1_SVM, color='blue', lw=2)
    plt.xlabel('Size')
    plt.legend(['f1_NB','f1_SVM'])
    plt.show()
    plt.figure()
    plt.plot(sample, average_auc_NB, color='orange', lw=2)
    plt.plot(sample,average_auc_SVM, color='blue', lw=2)
    plt.xlabel('Size')
    plt.legend(['auc_NB','auc_SVM'])
    plt.show()

def classify_duplicate(filename):
    data = pd.read_csv(filename)
    data.head()

    tfidf_vect = TfidfVectorizer(stop_words="english")

    docs = data.q1.values.tolist()+data.q2.values.tolist()
    docs_dtm = tfidf_vect.fit(docs)
    q1_dtm = tfidf_vect.transform(data['q1'])
    q2_dtm = tfidf_vect.transform(data['q2'])
    q1_dtm.shape
    q2_dtm.shape

    scores=[]
    for i in range(0,len(data)):
        sim_score = cosine_similarity(q1_dtm[i],q2_dtm[i])[0]
        bm25 = BM25Okapi([x.split(" ") for x in data["q1"].values.tolist()])
        tokenized_query = data.q2[i].split(" ")
        bm25_score = bm25.get_scores(tokenized_query)[i]
        scores.append([sim_score,bm25_score])
    scores

    clf_SVM =svm.LinearSVC()
    metrics = ["roc_auc"]
    cv_SVM = cross_validate(clf_SVM, scores,data['is_duplicate'], \
                                scoring=metrics, cv=5, \
                                return_train_score=True)

    return cv_SVM['test_roc_auc'].mean()

if __name__ == "__main__":
    print("Q1:")
    classify("train.csv","test.csv")
    print("Q2:")
    impact_of_sample_size("train_large.csv")
    print("\nQ3:")
    print(classify_duplicate("quora_duplicate_question_500.csv"))




