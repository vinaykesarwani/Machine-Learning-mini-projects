import numpy as np
import pandas as pd
import nltk

credits_df= pd.read_csv("Clustering\movies_recommendation\credits.csv")
movies_df=pd.read_csv("Clustering\movies_recommendation\movies.csv")

movies_df=movies_df.merge(credits_df, on='title')

movies_df=movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies_df.dropna(inplace=True)

import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df['genres']=movies_df['genres'].apply(convert)
movies_df['keywords']=movies_df['keywords'].apply(convert)

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter<3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies_df['cast']=movies_df['cast'].apply(convert3)

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L

movies_df['crew']=movies_df['crew'].apply(fetch_director)
movies_df['overview']=movies_df['overview'].apply(lambda x:x.split())
movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast']=movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['tags']=movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']+movies_df['crew']

new_df=movies_df[['movie_id', 'title', 'tags']]

new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()


from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')
