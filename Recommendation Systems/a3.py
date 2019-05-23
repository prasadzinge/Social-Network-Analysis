#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    
    movies["tokens"] = [tokenize_string(i) for i in movies["genres"]]
    
    return movies


# In[2]:


movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
movies = tokenize(movies)
movies['tokens'].tolist()


# In[3]:


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    
#     print (movies['tokens'])
    tok = defaultdict (lambda: 0)
    feature_list = []
    
    for i in movies['tokens']:
        f = defaultdict (lambda:0)
        for j in i:
            tok[j] +=1
            f[j] +=1
        feature_list.append(f)
#     print ('flist')
#     print (feature_list)
#     #print (tok)
    
    vocab = defaultdict (lambda: len(vocab))
    for i in sorted(tok):
        vocab[i]

#     print('vocab')
    vocab = dict(vocab)
#     print (vocab)
  
    dflen = movies.shape[0]
    csr_list = []
    for i in range(dflen):
        row=[]
        col=[]
        data=[]
        document= feature_list[i]
        max_k = document[max(document, key=document.get)]
        
        for token in document:
            if token in vocab:
                col.append(vocab[token])
                tfidf = (document[token]/max_k) * (math.log((dflen/tok[token]),10))
                data.append(tfidf)
        row = [0]*len(col)
        csr = csr_matrix((data,(row,col)),shape = (1, len(vocab)))
        csr_list.append(csr)
        
    feats = pd.DataFrame(csr_list,movies.index)
    movies['features'] = feats
    return movies,vocab


# In[4]:


featurize(movies)


# In[5]:


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    return float(np.dot(a, b.T).toarray()[0][0]/(np.linalg.norm(a.toarray())*np.linalg.norm(b.toarray())))


# In[6]:


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the reweighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
        
    prediction=[]
    
    #print (ratings_train)
    
    testset = ratings_test.iterrows()
    for ignore ,rated in testset:
        test_movie = movies.loc[movies.movieId == rated.movieId].squeeze()['features']
        user = ratings_train.loc[ratings_train.userId == rated.userId]
        
        weight = 0.
        cos = 0.
        neg_rating = True
        
        user_ratings = user.iterrows()
        for ign,m in user_ratings:
            
            train_movie = movies.loc[movies.movieId==m.movieId].squeeze()['features']
            cos_sim = cosine_sim(test_movie, train_movie)            
            if cos_sim > 0:
                cos += cos_sim
                weight += cos_sim * m.rating
                neg_rating = False
        if neg_rating:
            prediction.append(user.rating.mean())
        else:
            prediction.append(weight/cos)

    
    final = np.array(prediction)
    return final


# In[7]:


movies = pd.DataFrame([[123, 'horror|romance', ['horror', 'romance']],

                                [456, 'comedy|horror', ['comedy', 'horror']],

                                [789, 'horror', ['horror']],

                                [000, 'action', ['action']]],

                               columns=['movieId', 'genres', 'tokens'])

movies, vocab = featurize(movies)
ratings_train = pd.DataFrame([

                 [9, 123, 2.5, 1260759144],

                 [9, 456, 3.5, 1260759145],

                 [9, 789, 1, 1260759146],

                 [8, 123, 4.5, 1260759147],

                 [8, 456, 4, 1260759148],

                 [8, 789, 5, 1260759149],

                 [7, 123, 2, 1260759150],

                 [7, 456, 3, 1260759151]],

                                      columns=['userId', 'movieId', 'rating', 'timestamp'])

ratings_test = pd.DataFrame([

                 [7, 789, 4, 1260759152]],

                                     columns=['userId', 'movieId', 'rating', 'timestamp'])

 

round(make_predictions(movies, ratings_train, ratings_test)[0], 1)


# In[8]:


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


# In[9]:


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

