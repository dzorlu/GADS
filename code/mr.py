#!/usr/bin/env python
from operator import itemgetter

from mrjob.job import MRJob
from pymongo import MongoClient

class WordCount(MRJob):

    def mapper(self, _, line):

        # NOTE use counters to keep track of things (across nodes)
        self.increment_counter('mapper', 'lines', 1)
        
        words = line.split()
    
        for word in words:
            self.increment_counter('mapper', 'words', 1)
            yield word, 1

    def reducer(self, word, counts):
        self.increment_counter('reducer', 'words', 1)
        yield word, sum(counts)

class DistrGrep(MRJob):

    def mapper_init(self):
        self.PATTERN = 'Ishmael'

    def mapper(self, _, line):
        self.increment_counter('mapper', 'lines', 1)

        if self.PATTERN in line:
            yield line, 1

    # NOTE no reducer necessary here; what difference would including it
    # make in the output?

class InvertIndex(MRJob):

    def mapper(self, _, line):
        elts = line.rstrip().split(',')
        key, values = elts[0], elts[1:]
    
        for value in values:
            yield value, key

    def reducer(self, key, values):

        # NOTE why doesn't this work?
        # yield key, values

        self.increment_counter('reducer', type(values), 1)
        yield key, [v for v in values]

class DistrDB(MRJob):

    def mapper_init(self):

        # initialize db connection for mapper in mapper_init method!
        self.db = MongoClient()['test']
        self.mongo = self.db['bikes']

        # NOTE this drops test_coll at the top of each job...why is this
        # needed?
        self.mongo.drop()

    def mapper(self, _, line):
        fields = line.rstrip().split(',')
        num, addr, lat, lon = itemgetter(0, 2, 4, 5)(fields)

        if num == 'id':
            self.increment_counter('mapper', 'skipped lines', 1)
            return

        doc = {'_id': int(num), 'addr': addr, 'lonlat': map(float, (lon, lat))}
        self.mongo.insert(doc)

        self.increment_counter('mapper', 'processed lines', 1)

    # NOTE suppose this job had a reducer phase that required db access; would it be
    # necessary to re-initialize the connection?

class DoNothing(MRJob):
    # NOTE what will the output of this job look like?

    def mapper(self, _, line):
        yield line, 1

    def reducer(self, line, _):
        yield line, 1

if __name__ == '__main__':
    WordCount.run()
    # DistrGrep.run()
    # InvertIndex.run()
    # DistrDB.run()
    # DoNothing.run()
