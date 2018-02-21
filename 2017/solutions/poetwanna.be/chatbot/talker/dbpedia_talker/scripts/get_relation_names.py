TRIPLES_FILE = 'pagerank_en_2016-04_top_100000_triples.ttl'

PREFIXES = [
    '<http://www.w3.org/2000/01/rdf-schema#',
    '<http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    '<http://dbpedia.org/ontology/',
    '<http://dbpedia.org/property/',
    '<http://dbpedia.org/resource/',
    '<http://xmlns.com/foaf/0.1/',
    '<http://www.w3.org/2002/07/owl#',
    '<http://purl.org/dc/terms/',
    '<http://www.w3.org/2003/01/geo/wgs84_pos#',
    '<http://purl.org/dc/elements/1.1/',
    '<http://www.w3.org/2004/02/skos/core#',
    '<http://www.w3.org/ns/prov#'
]


relations = set()
no_prefix_for = set()

for triple in open(TRIPLES_FILE, 'r'):
    tokens = triple.split()
    subject, prop, value = tokens[0], tokens[1], ' '.join(tokens[2:-1])
    matched = False
    for prefix in PREFIXES:
        if prop.startswith(prefix):
            matched = True
            name = prop[len(prefix):].rstrip('>')
            if name not in relations:
                relations.add(name)
                # print "added", prop, name
            break
    if not matched and prop not in no_prefix_for:
        no_prefix_for.add(prop)
        print "Could not find prefix for %s" % prop


# print "FOUND:"
# for r in relations:
#     print r


print "NO PREFIX FOR:"
for p in no_prefix_for:
    print p

