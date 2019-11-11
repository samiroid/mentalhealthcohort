import codecs
from collections import defaultdict
import csv
import gzip
from ipdb import set_trace
import json
import os
import sys
# sys.path.append(ASMAT)
from ASMAT.lib.data import preprocess, shuffle_split

#input files
HOME="/Users/samir/Dev/resources/ASMAT_RAW_DATA/raw_datasets/mental_health"
PATH_TWEETS  = HOME+'/training_data/'
PATH_LABELS = HOME+"/anonymized_user_info_by_chunk_training.csv"
OUTPUT_PATH = 'DATA/processed/txt/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)        
#### --- TRAIN DATA ---- ####
tweets_by_user = {}
print("[reading user tweets]")
z=0
MAX_USERS=10
# MAX_USERS=float('inf')
MIN_TWEETS=100
for fname in os.listdir(PATH_TWEETS):	
	if os.path.splitext(PATH_TWEETS+fname)[1]!=".gz":
			print("ignored %s"% fname )
			continue			
	with gzip.open(PATH_TWEETS+fname, 'r') as f:			
		user = fname[:fname.index(".")] 		
		data = [preprocess(json.loads(l)['text']) for l in f]
		# data = [' '.join(json.loads(l)['text'].split()) for l in f]
		# data = set([json.loads(l)['text'] for l in f])
		if len(data) < MIN_TWEETS:
			print("\nignored user %s | %d tweets" % (user, len(data)))
			continue		
		tweets_by_user[user] = set(data)
		
	sys.stdout.write("\ruser: "+user+" ("+ str(z) +")"+" "*20)
	sys.stdout.flush()
	# if z>=MAX_USERS:
	# 	print("out early!!!!")
	# 	break	
	# else:
	# 	set_trace()
	z+=1

print("[writing user tweets]")
# user_corpus = codecs.open(OUTPUT_PATH+"mental_health_tweets","w","utf-8")
user_corpus = open(OUTPUT_PATH+"mental_health_tweets","w")
for user, twt in tweets_by_user.items():
	# set_trace()
	tweets = '\t'.join(twt)
	# tweets = ' '.join(tweets.split())
	#deal with encoding-decoding
	tweets = tweets.encode("utf-8","replace").decode("utf8")
	# set_trace()
	user_corpus.write("{}\t{}\n".format(user, tweets))

print("[writing tweets for word embedding training]")
# user_corpus = codecs.open(OUTPUT_PATH+"word_embeddings_corpus","w","utf-8")
user_corpus = open(OUTPUT_PATH+"word_embeddings_corpus","w")
for user, twt in tweets_by_user.items():
	# set_trace()
	tweets = '\n'.join(twt)	
	#deal with encoding-decoding
	tweets = tweets.encode("utf-8","replace").decode("utf8")
	user_corpus.write("{}\n".format(tweets))

print("[reading training labels]")
ptsd = {}
depression = {}
with open(PATH_LABELS) as fid:
	f = csv.reader(fid)
	next(f) #skip the header
	for r in f:
		user = r[0]
		cond = r[4]
		if cond == "ptsd":
			ptsd[user] = cond
		elif cond == "depression":
			depression[user] = cond
		elif cond == "control":
			ptsd[user] = cond
			depression[user] = cond
		
#stratified split
ptsd_tuples = [[x[1],x[0]] for x in ptsd.items()]
tmp_set, ptsd_test = shuffle_split(ptsd_tuples)
ptsd_train, ptsd_dev = shuffle_split(tmp_set)

depression_tuples = [[x[1],x[0]] for x in depression.items()]
tmp_set, depression_test = shuffle_split(depression_tuples)
depression_train, depression_dev = shuffle_split(tmp_set)

print("[writing PTSD data]")
with open(OUTPUT_PATH+"ptsd_train","w") as fod:
	for label, user in ptsd_train:		
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue
		fod.write("{}\t{}\n".format(label, user))
		
with open(OUTPUT_PATH+"ptsd_test","w") as fod:
	for label, user in ptsd_test:
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
with open(OUTPUT_PATH+"ptsd_dev","w") as fod:
	for label, user in ptsd_dev:		
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
print("[writing DEPRESSION data]")
with open(OUTPUT_PATH+"depression_train","w") as fod:
	for label, user in depression_train:		
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue
		fod.write("{}\t{}\n".format(label, user))
		
with open(OUTPUT_PATH+"depression_test","w") as fod:
	for label, user in depression_test:
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
with open(OUTPUT_PATH+"depression_dev","w") as fod:
	for label, user in depression_dev:		
		if user not in tweets_by_user: 
			# print("unknown dude %s" % user		)
			continue		
		fod.write("{}\t{}\n".format(label, user))
		