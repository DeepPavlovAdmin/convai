import os

RESOURCES_FILE = 'top_100000_resources.txt'
ALL_TRIPLES_DIR = '/pio/lscratch/1/qa_nips_data/dbpedia/2016-10/core/'
OUTPUT_TRIPLES_FILE = 'top_100000_triples.txt'

FILES_LIST = [
    'article_categories_en.ttl',
    'category_labels_en.ttl',
    'disambiguations_en.ttl',
    'homepages_en.ttl',
    'infobox_properties_en.ttl',
    'infobox_property_definitions_en.ttl',
    'instance_types_en.ttl',
    'instance_types_transitive_en.ttl',
    'interlanguage_links_chapters_en.ttl',
    'labels_en.ttl',
    'long_abstracts_en.ttl',
    'mappingbased_literals_en.ttl',
    'mappingbased_objects_en.ttl',
    'page_ids_en.ttl',
    'persondata_en.ttl',
    'revision_ids_en.ttl',
    'revision_uris_en.ttl',
    'short_abstracts_en.ttl',
    'skos_categories_en.ttl',
    'specific_mappingbased_properties_en.ttl'
]

RELATION_PREFIXES = [
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


def strip_subject(uri):
    return uri[len('<http://dbpedia.org/resource/'):].rstrip('>')


def strip_relation(uri):
    for prefix in RELATION_PREFIXES:
        if uri.startswith(prefix):
            return uri[len(prefix):].rstrip('>')


def upperfirst(x):
    return x[0].upper() + x[1:]


def strip_value(value):
    # value = value.encode('utf-8')
    if value.startswith('"'):
        return value.split('"^^')[0].lstrip('"')
    if value.startswith('<http://dbpedia.org/resource/'):
        return strip_subject(value)
        # return ' '.join(resource_name.split('_'))
    if value.endswith('"@en'):
        return value[:-len('"@en')].lstrip('"')
    # if '_' in value:
    #     return ' '.join(value.split('_'))
    return value


known_resources = set()
for line in open(RESOURCES_FILE):
    uri, pagerank = line.split()
    known_resources.add(uri)


output = open(OUTPUT_TRIPLES_FILE, 'w')

for filename in FILES_LIST:
    print "Processing file %s..." % filename
    for line in open(os.path.join(ALL_TRIPLES_DIR, filename), 'r'):
        if line.startswith('#'):
            continue
        tokens = line.split()
        subject, prop, value = tokens[0], tokens[1], ' '.join(tokens[2:-1])
        subject = strip_subject(subject)
        relation = strip_relation(prop)
        if subject in known_resources:
            output.write("%s|$|%s|$|%s\n" % (subject, relation, strip_value(value)))
        if value.startswith('<http://dbpedia.org/resource/'):
            value_resource = strip_subject(value)
            if value_resource in known_resources:
                indirect_relation = 'is' + upperfirst(relation) + 'Of'
                output.write("%s|$|%s|$|%s\n" % (value_resource, indirect_relation, subject))
    print "Done\n"

output.close()
