#!/usr/bin/env python

# line 1 is called a "shebang"...it's not a comment!
# this line tells the shell (eg your OS) how to run this script
# as an executable by specifying the "interpreter"

import json                         # import a couple things from python std library
from operator import itemgetter     # (look these up!)

CITIBIKE_FILE = 'citibike.json'     # global vars specifying input/output files
OUTPUT_FILE = 'citibike.csv'

def main(input_file=CITIBIKE_FILE, output_file=OUTPUT_FILE):            # define main func w/ default args
    """Parses json object from input file, creates output csv."""       # docstring...basic doc about this function

    with open(input_file, 'r') as f:        # open input file (using "with" context mgr)
        data = json.load(f)                 # load data from input file as json object

    stations = data['stationBeanList']      # extract stations data

    fields = ('id', 'stationName', 'stAddress1', 'stAddress2',          # tuple of fields to parse
        'latitude', 'longitude', 'availableBikes', 'availableDocks',
        'totalDocks', 'statusValue', 'statusKey', 'testStation')

    output_fields = ('id', 'name', 'addr1', 'addr2', 'lat', 'lon',      # tuple of output fields (for header row in output file)
        'bikes_avail', 'spots_avail', 'spots_total', 'status',
        'status_key', 'test_stn')

    with open(output_file, 'w') as g:                   # open output file
        g.write(','.join(output_fields) + '\n')         # write header row

        for stn in stations:                            # iterate over stations
            output = itemgetter(*fields)(stn)           # extract fields from stn...note tricky use of asterisk!
            g.write(','.join(map(str, output)) + '\n')  # write output to file


if __name__ == '__main__':      # this block specifies what to do if this script is run at "top-level" (as opposed to being imported by another script)
    main()                      # call main func (note no args specified, so run with default args)

# note: json.load() takes an open file handle (to a valid json file) as its argument
#       you can validate your json input using a tool such as www.jsonlint.com
