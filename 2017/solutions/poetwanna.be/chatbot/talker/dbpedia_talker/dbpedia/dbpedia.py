import cdb
import utils
from config import data_path
from os.path import join

PROPERTIES_DB_PATH = join(data_path, 'dbpedia/pagerank/top_100000_properties.cdb')
VALUES_DB_PATH = join(data_path, 'dbpedia/pagerank/top_100000_values.cdb')


class CDB(object):

    def __init__(self, db_path):
        print "CDB: opening", db_path
        self.db = cdb.init(db_path)
        self.cache = {}

    def getall(self, key):
        if key in self.cache:
            return self.cache[key]

        value = self.db.getall(key)
        try:
            if isinstance(value, str):
                value = unicode(value, 'utf8')
        except:
            pass
        if len(self.cache) > 1e3:
            self.cache = {}
        self.cache[key] = value

        return value


class DBPedia(utils.Singleton):

    def __init__(self):
        self.properties = CDB(PROPERTIES_DB_PATH)
        self.values = CDB(VALUES_DB_PATH)

    def get_properties(self, resource):
        return set(self.properties.getall(resource))

    def get_value(self, resource, prop):
        return set(self.values.getall(resource + '|$|' + prop))
