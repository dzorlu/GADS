import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as DT
from dateutil import parser

fin = open("tweets.json")
tweets = {}
i=0
for entry in fin:
	data = json.loads(entry)
	date = data['created_at']
	date = parser.parse(date).date()
	if date in tweets:
		tweets[date] += 1
	else:
		tweets[date] = 1

#Sort
tweets = sorted(tweets.iteritems(), key=operator.itemgetter(0))

output=[]
for line in tweets:
	output.append({'date':line[0].strftime('%m/%d/%Y'),'count':line[1]})

#Save
pd.DataFrame(output).to_csv('tweet_count.csv',index=None)
