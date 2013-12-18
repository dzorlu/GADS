import pymongo
import sys

host = "mongodb://localhost"


def main():

  '''connect to the database'''
  connection = pymongo.MongoClient(host)

  '''connect to the collection'''
  db = connection.test
  people = db.

  try:
    '''Python dictionary'''
    person = {'name':'Charlie Brown', 'role':'Lovable Loser',
              'address': {'street': '110 High Street',
                          'state' : 'MN',
                          'city': 'Minneapolis'},
              'interests':['flying', 'dogs', 'Peanuts'],
              'best_friend':'Linus van Pelt',
              'dog': 'Snoopy'}
              #'_id': 312038102}

    people.insert(person)
    print person['name'], ' inserted'
  except:
    print "insert failed:", sys.exc_info()[0]

if __name__ == "__main__":
  main()