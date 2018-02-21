import cdb

TRIPLES_FILE_PATH = 'top_100000_triples.txt'
OUTPUT_PROPERTIES_DB = 'top_100000_properties.cdb'
OUTPUT_VALUES_DB = 'top_100000_values.cdb'


properties_db = cdb.cdbmake(OUTPUT_PROPERTIES_DB, OUTPUT_PROPERTIES_DB + '.tmp')
values_db = cdb.cdbmake(OUTPUT_VALUES_DB, OUTPUT_VALUES_DB + '.tmp')

print "Loading DBPedia triples..."
for triple in open(TRIPLES_FILE_PATH, 'r'):
    subject, prop, value = triple.split('|$|')
    properties_db.add(subject, prop)
    values_db.add(subject + '|$|' + prop, value.rstrip('\n'))
print "Done"

properties_db.finish()
print "Properties cdb created"

values_db.finish()
print "Values cdb created"
