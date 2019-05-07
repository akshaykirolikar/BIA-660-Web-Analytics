
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import LatentDirichletAllocation


def cluster_kmean(train_file,test_file):
    f_train = open(train_file,encoding="utf-8")
    train_data = json.load(f_train)
    df_train = pd.DataFrame(train_data,columns=['text'])
    f_train.close()


    f_test = open(test_file,encoding='utf-8')
    test_data = json.load(f_test)
    df_test = pd.DataFrame(test_data,columns=['text','labels'])
    f_test.close()
    
    labels = df_test.labels
    labels = list(set(sum(labels,[])))[:3]
    
    

    tfidf_vect = TfidfVectorizer(stop_words="english",min_df=5) 

    dtm = tfidf_vect.fit_transform(df_train['text'])

    num_clusters=3
    
    clusterer = KMeansClusterer(num_clusters, cosine_distance,repeats=20)
    
    clusters = clusterer.cluster(dtm.toarray(),assign_clusters=True)

    
    centroids=np.array(clusterer.means())

    sorted_centroids = centroids.argsort()[:, ::-1] 

    voc_lookup= tfidf_vect.get_feature_names()

    test_dtm = tfidf_vect.transform(df_test.text)

    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]


    df_test['label_test'] = df_test['labels'].apply(lambda x : x[0])
    

    confusion_df = pd.DataFrame(list(zip(df_test["label_test"].values, predicted)),columns = ["actual_class", "cluster"])
    

    df_result = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.actual_class)
    
    print(df_result)
    
    df_clusterLabelsPredicted = list(df_result.apply(lambda x: x.idxmax(), axis=1))
    cluster_dict = dict((i,j) for i,j in enumerate(df_clusterLabelsPredicted))
    
    predicted_target=[cluster_dict[i] for i in predicted]

    print(metrics.classification_report (df_test["label_test"], predicted_target))
    for i in cluster_dict:
        print("Cluster %d : Topic %s" % (i,cluster_dict[i]))


def cluster_lda(train_file,test_file):
    f_train = open(train_file,encoding="utf-8")
    train_data = json.load(f_train)
    df_train = pd.DataFrame(train_data,columns=['text'])
    f_train.close()

    f_test = open(test_file,encoding='utf-8')
    test_data = json.load(f_test)
    df_test = pd.DataFrame(test_data,columns=['text','labels'])
    f_test.close()
    

    text_train = df_train.text
    tf_vectorizer = CountVectorizer(max_df = 0.90 , min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(text_train)
    
    num_topics = 3

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=25,verbose=1,
                                    evaluate_every=1, n_jobs=1,
                                    random_state=0).fit(tf)

   
    text_test = df_test.text
    tf = tf_vectorizer.transform(text_test)
    tf_feature_names = tf_vectorizer.get_feature_names()

    df_test['label_test'] = df_test['labels'].apply(lambda x : x[0])

    topic_assign=lda.transform(tf)

    predicted = np.argmax(topic_assign,axis=1)

    confusion_df = pd.DataFrame(list(zip(df_test["label_test"].values, predicted)), columns = ["actual_class", "cluster"])

    df_result = pd.crosstab(index=confusion_df.cluster, columns=confusion_df.actual_class)
    print(df_result)
    
    df_predictedClusterLabels = list(df_result.apply(lambda x: x.idxmax(), axis=1))
    cluster_dict = dict((i,j) for i,j in enumerate(df_predictedClusterLabels))
    predicted_target=[cluster_dict[i] for i in predicted]

    print(metrics.classification_report (df_test["label_test"], predicted_target))
    for i in cluster_dict:
        print("Cluster %d : Topic %s" % (i,cluster_dict[i]))


if __name__ == "__main__":
    print("Q1:")
    cluster_kmean("train_text.json","test_text.json")
    print("\n")

    print("Q2:")
    cluster_lda("train_text.json","test_text.json")

