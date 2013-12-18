#!/usr/bin/env python

import json
import pymongo
import urllib2
import sys
import datetime
from pandas import Period, DataFrame, Period, concat
from dateutil import parser
from datetime import datetime, timedelta
import calendar
import getopt


# establish a connection to the database.
host = "mongodb://localhost" 
connection = pymongo.Connection(host)
users = connection.test.users
mentions = connection.test.mentions


def twitter_daily_aggregate(retrievaldate):

	#Date Retrieval
	d=[]
	dt = parser.parse(retrievaldate) + timedelta(days=-1)
	d.append(dt)
	d.append(d[-1] + timedelta(days=1))

	#DataFrame Init
	ctrend = DataFrame()
	while d[-1] < datetime.utcnow(): 
		print 'processing ', d[-1], ' ..........'
		#Daily Mention Count
		mnts = twitter_count(d, mentions)

		#User Follower Count
		usrs =  twitter_follower(d,users)
		#Join
		trend = mnts.join(usrs)
		trend['Date'] = Period(d[-1],'D')
		#Append to DataFrame
		ctrend = concat([ctrend,trend])
		#Extend Dates
		d.append(d[-1] + timedelta(days=1))
	#Join DataFrames and Fill NAs
	ctrend =  ctrend.fillna(0)
	#Save
	print 'printing the file'
	ctrend.to_csv('twitter_trend.csv')
	return ctrend

def twitter_follower(d, collection = users):
	#MongoDB Aggregate Query - Daily Follower Count for each User defined by CDPID. 
	#Creating the upper and lower boundary for the query. 
	upper_bound_start_ts = float(calendar.timegm(d[-1].utctimetuple())*1000); 
	upper_bound_end = d[-1] + timedelta(days=1); 
	upper_bound_end_ts = float(calendar.timegm(upper_bound_end.utctimetuple())*1000)
	#MongoDB 
	users = collection.aggregate([{'$match':{'timestamp':{"$gt": upper_bound_start_ts, "$lt": upper_bound_end_ts}}},
		                {'$project':{'cdpid':1,'screen_name':1,'followers_count':1,'friends_count':1}},
		                #Average by Screenname if the same user name is retrieved twice
		                {'$group': {'_id': '$screen_name', 'friends_count': {'$avg':'$friends_count'},'followers_count': {'$avg':'$followers_count'},'cdpid':{'$addToSet':'$cdpid'}}},
		                {'$unwind':'$cdpid'},
		                {'$group': {'_id': '$cdpid', 'friends': {'$sum':'$friends_count'},'followers': {'$sum':'$followers_count'}}}])
	#Followers
	users = DataFrame(users['result'])
	users.index = users._id;  users=users.drop('_id',axis=1); users = users.sort_index();
	print 'Friends and Followers for ', d[-1], ' processed'
	return(users)

def twitter_count(d,collection = mentions):

	#MongoDB Query - Mentions
	#The Day Of
	upper_bound_start_ts = float(calendar.timegm(d[-1].utctimetuple())*1000); 
	upper_bound_end = d[-1] + timedelta(days=1); 
	upper_bound_end_ts = float(calendar.timegm(upper_bound_end.utctimetuple())*1000)
	
	# #Retrieve Tweeets that are not authored by the user itself. 
	mentions = 	collection.aggregate([
							{'$match': {'timestamp':{'$gt': upper_bound_start_ts, '$lt': upper_bound_end_ts}}},
							{'$unwind':'$cdpid'},
							{'$group':{'_id':'$cdpid','mentions':{'$sum':1}}}])
	#Tweets collection does not need unwind unlike mentions collection. 

	mentions = DataFrame(mentions['result']); 
	mentions.index = mentions._id;  mentions=mentions.drop('_id',axis=1); mentions= mentions.sort_index();
	#mts['Date'] = Period(d[-2],'D')
	print  d[-1], ' processed'
	return(mentions)

if __name__ == "__main__":

	try:
		opts,args = getopt.getopt(sys.argv[1:],'d:', ['date='])
	except getopt.GetoptError:
		pass
	#Parse Arguments
	for opt, arg in opts:
		if opt in ('-d','--date'):
			retrievaldate = arg
		else:
			print 'input not recognized'
			break

	ctrend = twitter_daily_aggregate(retrievaldate)






