#!/usr/bin/env python

import json
import pymongo
import urllib2
import sys
import datetime
from pprint import pprint

# establish a connection to the database. 
connection = pymongo.Connection("mongodb://localhost")

# #Get a handle on peanuts collection
# peanuts = connection.test.peanuts 
#Get a handle on the grades collection.
grades = connection.test.grades
#Get a handle on students collection
# students = connection.test.students
#Get a handle on things collection.
things = connection.test.things
#Get a handle to the reddit database
stories=connection.test.stories
LINK = 'http://www.reddit.com/.json'


def insert(link = LINK):
	#Reddit and JSON
	reddit_page = urllib2.urlopen(link)
	#Load
	'''Json.loads decodes the JSON file into a Python dictionary'''
	parsed = json.loads(reddit_page.read())
	#Insert into Mongo
	for item in parsed['data']['children']: 
		stories.insert(item['data'])
		print item['data']['title']
		# print 'Story: {0} with score:  {1} inserted'.\
		# 	item['data']['title'].encode('utf8'), item['data']['score'])
	stories.drop()

   
def find():

	query = {'type':'exam'}

	#Sort, Skip, Limit 
	'''
	Sort, Skip, Limit. Calls happen in this order. Database really run the query until you start
	iterating through the cursor. The query is postponed. 
	Notice that the synthax for sorting in the driver is different. Here, Sort works with tuples
	'''

	cursor = grades.find(query)
	cursor = cursor.sort([('score',pymongo.DESCENDING)])
	cursor = cursor.limit(5)

	print cursor.count(), ' documents found'

	for doc in cursor: print doc

def find_operators():

	'''
	Query Operators:
		http://docs.mongodb.org/manual/reference/operator/query/
	#array operators
		$push --> add to the end
		$pop --> pop from the beginning -1
		$pull --> remove the matching items
	'''
	#$gt and $lt
	query = {'type':'exam', 'score':{'$gt':50, '$lt':70}}

	cursor = grades.find(query)
	print cursor.count(), ' documents found'

	for doc in cursor: print doc
	


def update():

	print 'updating the document ..'

	#Wholesale update
	query = {'student_id':0, 'type':'exam'}
	
	student = grades.find_one(query)
	print "before: ", student
	#Add Review Date
	student['review_date'] = datetime.datetime.utcnow()
	# update the record with update. Note that there an _id but DB is ok with that
	# because it matches what was there.
	grades.update(query, student)
	student = grades.find_one(query)
	print "after: ", student

	#Remove Review Dates
	remove_review_date()


def update_set():

	#Selective Update
	query = {'student_id':0, 'type':'exam'}

	print "updating record using set"

	# get the doc
	student = grades.find_one(query)
	print "before: ", student

	# update using set
	grades.update(query,
	              {'$set':{'review_date':datetime.datetime.utcnow()}})

	score = grades.find_one(query)
	print "after: ", score

	#Remove Review Dates
	remove_review_date()


def upsert():

	print "updating with upsert"

	try:
	    # lets remove all stuff from things
	    things.drop()

	    things.update({'thing':'apple'}, {'$set':{'color':'red'}}, upsert=True)
	    things.update({'thing':'pear'}, {'color':'green'}, upsert=True) #Strange that 'pear' is missing!
	    #things.update({'thing':'pear'}, {'$set':{'color':'green'}}, upsert=True)

	    apple = things.find_one({'thing':'apple'})
	    print "apple: ", apple
	    pear = things.find_one({'thing':'pear'})
	    print "pear: ", pear

	except:
	    print "Unexpected error:", sys.exc_info()[0]
	    raise

def remove_review_date():
		#Utility function to keep grades collection clean. 
	print "\n\nremoving all review dates"

	try:
	    grades.update({},{'$unset':{'review_date':1}},multi=True)

	except:
	    print "Unexpected error:", sys.exc_info()[0]
	    raise


def remove():
	'''
	Works like find. If you specify an argument, then the only matching document will be removed
	The difference is an implementation detail. Remove is one-by-one update. Dropping is faster. 
	Metadata remains in existence with remove. Multiple remove operations are not atomic. 
	'''

if __name__ == '__main__':
	#insert(link = LINK)
	#find()
	#find_operators()
	#update()
	#update_set()
	upsert()
