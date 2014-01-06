#!/usr/bin/env python
from operator import itemgetter

from redis import StrictRedis       # this is the redis api for python (install
                                    # via "pip install redis")

INPUT_FILE = 'citibike.csv'
def do_writes(input_file=INPUT_FILE):
    """CREATE"""
    db = StrictRedis()

    # NOTE pipelining (eg MULTI/EXEC) permits atomic batched transactions
    db_pipe = db.pipeline()

    print 'populating db...'
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):

            if i == 0:
                continue

            fields = line.rstrip().split(',')
            redis_id, addr1, lat, lon = fields = itemgetter(0, 2, 4, 5)(fields)

            # set multiple elts of hash at once with hmset
            # db.hmset(redis_id, {'addr': addr1, 'lon': lon, 'lat': lat})
            db_pipe.hmset(redis_id, {'addr': addr1, 'lon': lon, 'lat': lat})

    # NOTE pipelined commands need to be executed!
    db_pipe.execute()

def do_reads():
    """READ"""
    db = StrictRedis()

    read_keys = (128, 157, 216, 236, 248)

    for key in interesting_keys:
        rec = db.hgetall(key)
        print rec

def do_updates():
    """UPDATE"""
    db = StrictRedis()
    db_pipe = db.pipeline()

    update_keys = (260, 271, 285, 300, 310)

    print 'before...'
    for key in update_keys:
        print db.hgetall(key)
        db_pipe.hset(key, 'city', 'NYC')

    db_pipe.execute()

    print '\nafter...'
    for key in update_keys:
        print db.hgetall(key)
        

def do_deletes():
    """DELETE"""
    db_pipe = StrictRedis().pipeline()

    print 'performing a few deletes...'
    del_keys = (260, 271, 285, 300, 310)
    for key in del_keys:
        db_pipe.delete(key)

    print 'db_pipe.command_stack:\n{}'.format(db_pipe.command_stack)
    db_pipe.execute()
    
if __name__ == '__main__':
    do_writes()
    do_reads()
    # do_updates()
    # do_deletes()
