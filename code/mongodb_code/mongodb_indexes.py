#!/usr/bin/env python

import json
import pymongo
import urllib2
import sys
import datetime
import timeit
from pprint import pprint

# establish a connection to the database. 
connection = pymongo.Connection("mongodb://localhost")

#Get a handle on students collection
students = connection.test.students
foo = connection.test.foo


def index_explain():
	#Drop all indices
	foo.drop_indexes()

	#Create an index
	foo.ensure_index([('a',1),('b',1),('c',1)])
	print 'index created\n'

	#A query where database cannot use the findex
	#Tagging 'explain' we get a document explaining the method
	print '\nNo index \n'
	pprint(foo.find({'c':1}).explain())

	#cursor = foo.find({'c':1})
	#for iter in cursor : print iter

	#A query where index will be used.
	print '\nIndex used \n'
	pprint(foo.find({'a':1}).explain())

	#cursor = foo.find({'a':1})
	#for iter in cursor : print iter

	#A query where index can be only used in sort.
	#Recreate another index to illustrate the next point
	foo.drop_index([('a',1),('b',1),('c',1)])
	foo.ensure_index([('a',1),('b',1)])
	print '\nIndex will be used to sort]]\n'
	pprint(foo.find({'$and':[{'c':{'$gt':250},'c':{'$lt':500}}]}).\
		sort([('a',1)]).explain()) #Index used for sorting
	
	# cursor = foo.find({'$and':[{'c':{'$gt':250},'c':{'$lt':500}}]}).\
	# 	sort([('a',1)])
	# for iter in cursor : print iter


if __name__ == '__main__':
	index_explain()

