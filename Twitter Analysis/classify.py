#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

# In[9]:


from collections import Counter,defaultdict
import requests
import pickle
import re
from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np 
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def get_names_list():
    males = requests.get('https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    """checking for amiguous names and using the ones which have  """
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])    
    return male_names, female_names

def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):

    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens

def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):

    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def get_tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):

    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def make_feature_matrix(tokens_list, vocabulary,tweets):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()
        
def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 'female'
    elif name in male_names:
        return 'male'
    else:
        return 'unknown'
    
def download_afinn():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    return afinn

def afinn_sentiment(terms, afinn):
    p = 0
    n = 0
    for t in terms:
        if t in afinn:
            if afinn[t] > 0:
                p += afinn[t]
            else:
                n += -1 * afinn[t]
    return p, n

def tokenize1(tweets):
    return re.sub('\W+', ' ', tweets.lower()).split()

def check_polatiry(tweets, afinn):
    positives = 0
    negatives = 0
    positive_tweets = []
    negative_tweets = []
    tokens = [tokenize1(t) for t in tweets]
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment(token_list, afinn)
        if pos > neg:
            positive_tweets.append(tweet)
        elif neg > pos:
            negative_tweets.append(tweet)
            negatives +=1
    pos_pct = (len(positive_tweets)/len(tweets))*100
    neg_pct = (len(negative_tweets)/len(tweets))*100
    
    return (positive_tweets, negative_tweets)

def cross_validation(X, y, nfolds):
    
    """ Compute average cross-validation acccuracy."""
    cv = KFold(n_splits=nfolds)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    print(np.std(accuracies))
    print(accuracies)
    return avg

def main():
    print('Lets do Sentiment Analysis Based on Afinn')
    afinn = download_afinn()
    tweets = pickle.load(open('tweets.pkl', 'rb'))
    tweets = [t for t in tweets]
    print('Found %d Tweets' % (len(tweets)))
    
    
    '''Getting The list of males and females'''
    male_names, female_names = get_names_list()
    print('Number of Males: %.0f, Number of Females: %.0f' %(len(male_names), len(female_names)))
    
    
    tokens_list = [get_tokens(t, use_descr=True, lowercase=True,
                            keep_punctuation=False, descr_prefix='d=',
                            collapse_urls=False, collapse_mentions=False)
              for t in tweets]
    males = []
    females = []
    unknowns = []
    mname = []
    fname = []
    uname = []
    for t in tweets:
        n=get_first_name(t)
        if get_gender(t, male_names, female_names) == 'male':
            males.append(t['text'])
            mname.append(n)
        elif get_gender(t, male_names, female_names) == 'female':
            females.append(t['text'])
            fname.append(n)
        elif get_gender(t, male_names, female_names) == 'unknown':
            unknowns.append(t['text'])
            uname.append(n)
    
    file = open("classify.txt","w+",encoding='utf-8')
    file.write('\nResults from Classify.py\n')
    file.write('\nmales : %d' %len(males))
    file.write('\nfemales : %d' %len(females))
    file.write('\nunknowns : %d' %len(unknowns))
    
    print('males : %d' %len(males))
    print('females : %d' %len(females))
    print('unknowns : %d' %len(unknowns))
    
    mp,mn = check_polatiry(males,afinn)
    fp,fn = check_polatiry(females,afinn)
    up,un = check_polatiry(unknowns,afinn)

    print ('Males have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(mp)/len(males)*100,len(mn)/len(males)*100))
    file.write ('\nMales have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(mp)/len(males)*100,len(mn)/len(males)*100))
    
    print ('Females have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(fp)/len(females)*100,len(fn)/len(females)*100))
    file.write ('\nFemales have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(fp)/len(females)*100,len(fn)/len(females)*100))
    
    print ('Unknown have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(up)/len(unknowns)*100,len(un)/len(unknowns)*100))
    file.write ('\nUnknown have %0.2f %% positive tweets and %0.2f %% Negative tweets' %(len(up)/len(unknowns)*100,len(un)/len(unknowns)*100))
    
    file.write('\n\nExamples from Each Class: \nMale: ')
    file.write(mname[0])
    file.write('\nFemale: ')
    file.write(fname[0])
    file.write('\nunknown: ')
    file.write(uname[0])
    
    
    file.write('\n\nMales Positive Tweet Example:\n')
    file.write(mp[0])
    file.write('\n\nMales Negative Tweet Example:\n')
    file.write(mn[0])
    file.write('\n\nFemales Positive Tweet Example:\n')
    file.write(fp[0])
    file.write('\n\nFemales Negative Tweet Example:\n')
    file.write(fn[0])
    file.write('\n\nUnknown gender Positive Tweet Example:\n')
    file.write(up[0])
    file.write('\n\nUnknown gender Negative Tweet Example:\n')
    file.write(un[0])
    file.close()

    
    vocabulary = defaultdict(lambda: len(vocabulary))
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  

    X = make_feature_matrix(tokens_list, vocabulary,tweets)
    print('shape of X:', X.shape)
    y = np.array([get_gender(t, male_names, female_names) for t in tweets])
    print('gender labels:', Counter(y))
    print('avg accuracy', cross_validation(X, y, 5))    
if __name__ == "__main__":
    main()

