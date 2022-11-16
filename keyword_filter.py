import os
import json
 
def parsing(word):
    word = word[:-1]
    return word.rstrip()
 
def parsing_(word):
    word = word[:-5]
    return word.rstrip()
 
def parsing_2(word):
    word = word[2:-1]
    return word.rstrip()
 
def read_keyword():
    keyword_f = open(path+'/final_data/RecordedFuture_industry_term_2.txt', 'r')
 
    lines = keyword_f.readlines()
    keywords = lines[0].split(',')
    keywords = list(map(parsing_2, keywords))
    keywords[0] = 'via'
    keywords = keywords[:-1]
    return keywords
 
path = '/home/'
keywords = read_keyword()
keywords.remove('via')
user_names = os.listdir(path+'/final_data/training_data/top_100/')
 
for i, user_name in enumerate(user_names):
    with open(path+"/final_data/test_data/cyber/{}.json".format(user_name), "r") as tweet_f:
        with open(path+"/final_data/filter/{}.json".format(user_name), "w") as filtered_f:
            if not tweet_f:
                continue
            try:
                tweets = [json.loads(line) for line in tweet_f]
 
                if not tweets:
                    continue
                for tweet in tweets:
                    for k in keywords:
                        if k in tweet['text']:
                            json.dump(tweet, filtered_f)
                            filtered_f.write('\n')
                            break
            except:
                continue
