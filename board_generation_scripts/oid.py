#!/usr/bin/env python3

# using pymongo-2.2
from bson.objectid import ObjectId
import datetime
import sys
from dateutil.tz import tzutc, tzlocal
import time


def usage():
    print("example: oid.py [DATE]", file=sys.stderr)
    print("\t\t\tDATE should be in in YYYYMMDD format", file=sys.stderr)
    print("\t\t\tIf DATE is not provided, the whole dataset will be saved.", file=sys.stderr)
    print("\nDEPRECATED: oid.py <competition day (1-7 or 0 for all days)>", file=sys.stderr)
    sys.exit(1)


deadlines = [
    (datetime.datetime(2017, 7, 24, 0, 0, 0, 0, tzlocal()), datetime.datetime(2020, 1, 1, 0, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 24, 0, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 24, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 24, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 25, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 25, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 26, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 26, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 27, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 27, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 28, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 28, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 29, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 29, 22, 0, 0, 0, tzlocal()), datetime.datetime(2017, 7, 30, 22, 0, 0, 0, tzlocal())),
    (datetime.datetime(2017, 7, 30, 22, 0, 0, 0, tzlocal()), datetime.datetime(2020, 1, 1, 1, 0, 0, 0, tzlocal()))

]

if len(sys.argv) > 2:
    usage()
elif len(sys.argv) == 1:
    inp = "0"
else:
    inp = sys.argv[1]

if len(inp) == 1:
    try:
        day = int(inp)
    except ValueError:
        usage()
    if 0 <= day < len(deadlines):
        start_date = deadlines[day][0]
        end_date = deadlines[day][1]

        oid_start = ObjectId.from_datetime(start_date.astimezone(tzutc()))
        oid_stop = ObjectId.from_datetime(end_date.astimezone(tzutc()))
    else:
        usage()
else:
    try:
        d = datetime.datetime(*time.strptime(sys.argv[1], "%Y%m%d")[:6], tzinfo=tzlocal())
        oid_stop = ObjectId.from_datetime(d + datetime.timedelta(hours=22))
        oid_start = ObjectId.from_datetime(d - datetime.timedelta(hours=2))
    except ValueError:
        usage()

print('{ "_id" : { "$gte" : { "$oid": "%s" }, "$lt" : { "$oid": "%s" } } }' % (str(oid_start), str(oid_stop)))
