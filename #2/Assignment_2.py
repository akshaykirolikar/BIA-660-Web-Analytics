
import pandas as pd
import numpy as np


def analyze_tf_idf(arr,K):
    arr.shape
    doc_len = np.sum(arr,axis=1) 
    doc_len = doc_len.reshape(1,np.size(doc_len))
    tf = arr.T/doc_len
    tf = tf.T
    tf
    df = np.where(arr>0,1,0)
    df = np.sum(df,axis=0)
    df
    tf_idf = tf/(np.log(df)+1)
    tf_idf
    tmp=np.argsort(-tf_idf)
    top_K = tmp[:,:K]
    return tf_idf,top_K

filepath='./question.csv'

def analyze_data(filepath):
    df = pd.read_csv(filepath)
    df1=df.loc[df['answercount']>0]
    df1=df1.sort_values(by=['viewcount'],ascending=False)
    df1=df1[['title','viewcount']]
    df1=df1.head(3)
    df2=df.groupby(by='quest_name').size()
    df2=df2.sort_values(ascending=False)
    df2=df2.to_frame()
    df2=df2.rename(columns={0 :'count'})
    print(df2.head(5))
    print('\n')
    df3=df.copy()
    first_tag = df['tags'].apply(lambda x: x.split(',')[0])
    df3=df3[['viewcount','answercount','tags']]
    df4=df3[['tags','viewcount']]
    df4.insert(0,'first_tag',first_tag)
    df4=df4.set_index('first_tag')
    df4=df4.loc[['python','pandas','dataframe']]
    print("Mean viewcount :")
    print(df4.groupby(by=df4.index).mean())
    print('\n')
    print("Max viewcount :")
    print(df4.groupby(by=df4.index).max())
    print('\n')
    print("Min viewcount :")
    print(df4.groupby(by=df4.index).min())
    print('\n')
    df_crosstab=df.copy()
    df_crosstab=df_crosstab[['answercount','tags']]
    df_crosstab['first_tag']=df_crosstab['tags'].apply(lambda x : x.split(',')[0])
    df_crosstab=pd.crosstab(index=df_crosstab['answercount'],columns=df_crosstab['first_tag'])
    print("Crosstable :\n")
    print(df_crosstab)
    print('\n')
    print('Number of "python" questions never answered :\n')
    print(df_crosstab.loc[[0],['python']])
    print('\n')
    print('Number of "python" questions answered once:\n')
    print(df_crosstab.loc[[1],['python']])
    print('\n')
    print('Number of "python" questions answered twice:\n')
    print(df_crosstab.loc[[2],['python']])
    print('\n')
    return

if __name__ == "__main__":  
    
    # Test Question 1
    arr=np.array([[0,1,0,2,0,1],[1,0,1,1,2,0],[0,0,2,0,0,1]])
    
    print("\nQ1")
    tf_idf, top_k=analyze_tf_idf(arr,3)
    print(top_k)
    
    print("\nQ2")
    analyze_data('question.csv')
    
    # test question 3
    '''print("\nQ3")
    analyze_corpus('question.csv')'''