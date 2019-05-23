#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8


from collections import Counter, defaultdict
import matplotlib.pyplot as plt
#import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle

consumer_key = 'ykwCWh33spQ6agbDNE7DpS0xT'
consumer_secret = 'z41yHzweia7EMAJCirQZHoVYf30wILiHZGpkEMg2mm7SflkVbq'
access_token = '557903245-E24svdJftjMFtSl0JUrgkFwvlkhXZvZnDHR4hNqz'
access_token_secret = 'bzCGPjWjtmz76SvW8RPmOkckbMU8wZElcTUr4qAPnkNh6'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    user_objects = robust_request(twitter, "users/lookup", {'screen_name':screen_names})
    return user_objects

def get_users_name(twitter, ids):
    user_objects = robust_request(twitter, "users/lookup", {'user_id':ids})
    for u in user_objects:
        user_name = u['screen_name']
    return user_name

def get_friends(twitter, screen_names):
    '''Get friends using Screen Name'''
    friends_name = []
    friends_id = robust_request(twitter, "friends/ids", {'screen_name':screen_names,'count':150})
    for f in friends_id:
        friends_name.append(get_users_name(twitter,f))
    return friends_name

def write_friends(users):
    file = open("users.txt","w+")
    for u in users:
        for f in u['friends']:
            file.write(u['screen_name']+":"+f+"\n")
    file.close()       

def get_tweets(twitter, track, limit):
    '''get user object using screen name'''
    tweet = []
    tweets = robust_request(twitter, 'statuses/filter', {'track':track,'language':'en','locations':'-122.75,36.8,-121.75,37.8,-74,40,-73,41'})
    for t in tweets:
        tweet.append(t)
        if len(tweet) >= limit:
            break
    return tweet


# In[2]:


def main():
    screen_names =['paulpogba','MarcusRashford','JesseLingard','AnthonyMartial','LukeShaw23','AnderHerrera','D_DeGea']
    twitter = get_twitter()
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    num_users = 0
    
    for u in users:
        u['friends']=get_friends(twitter, u['screen_name'])
        num_users+=len(u['friends'])
        
    write_friends(users)
    print('User and Friends data stored in users.txt')
    
    tweets = get_tweets(get_twitter(),'manutd',1000)
    pickle.dump(tweets, open('tweets.pkl','wb'))
    
    print('Tweets Stored in tweets.pkl')
    
    file = open('collect-summary.txt','w+')
    file.write('\nNumber of  Users collected: %d' %num_users)
    file.write('\nNumber of  messages collected: %d' %len(tweets))
    file.close()
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




