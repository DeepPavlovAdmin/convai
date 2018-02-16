INPUT_FILE_NAME = 'pagerank_en_2016-04_top_100000.ttl'
OUTPUT_FILE_NAME = 'pagerank_en_2016-04_top_100000.txt'


def get_resource_name(uri):
    prefix = '<http://dbpedia.org/resource/'
    return uri[len(prefix):].rstrip('>')


def get_pagerank_value(pagerank):
    suffix = '"^^<http://www.w3.org/2001/XMLSchema#float>]'
    return pagerank.lstrip('"')[:-len(suffix)]


with open(OUTPUT_FILE_NAME, 'w') as f:
    for line in open(INPUT_FILE_NAME, 'r'):
        uri, _, _, pagerank, _ = line.split()
        f.write('%s %s\n' % (get_resource_name(uri), get_pagerank_value(pagerank)))
