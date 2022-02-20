import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#diavazoume ta datasetes
df=pd.read_csv("JobsDataset.csv")
df2=pd.read_csv("JobsDataset_Testing.csv")


features = ['Query','Job Title','Description']
#pairnoume tis arxikes peigrafes
X = df[features]
#pairnoume gia kathe perigrafi to y
y= df['Query']
#diavazoume apo to test synolo tis perigrafes
x2=df['Description']
x3=df2['Description']
#pairnoume tis katigories se mia lista
catg=list(y.drop_duplicates())

#pairnoume mono ta tokens apo ta keimena
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
#pairnoume ena dianisma me to poio token aneikei se kathe keimeno
X_train_counts = count_vect.fit_transform(x2)
X_train_counts.shape

#painroume tous diktew Tf tf-idf 
from sklearn.feature_extraction.text import TfidfTransformer


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y)


X_new_counts = count_vect.transform(x3)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
txt=list(x3)
for i in range(len(predicted)):
    print (i, predicted[i] )

