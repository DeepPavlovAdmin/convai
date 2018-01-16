'''
ceated on Oct 27, 2017

@author: CheongAn Lee
'''

import os
import sys
import progressbar
import time
import resource

from whoosh.analysis.analyzers import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh import qparser
from whoosh import scoring

import whoosh.index as index

from zimply import ZIMFile
from html.parser import HTMLParser




class DRHtmlParser(HTMLParser):
    def __init__(self):
        self.output = []
        self.body = False
        self.p = False
        self.pidx = 0
        super().__init__()
    
    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.body = True
        elif tag == "p":
            self.p = True

    def handle_endtag(self, tag):
        if tag == "body":
            self.body = False
        elif tag == "p":
            self.p = False
            self.pidx += 1

    def record(self):
        return self.body and self.p
    
    def handle_data(self, data):
        if self.record():
            try:
                self.output[self.pidx] += " " + data
            except IndexError:
                self.output.append(data)
        
    def __enter__(self):
        return self
    
    def __exit__(self, t, value, traceback):  # @UnusedVariable
        self.close()


class TitleParser(HTMLParser):
    def __init__(self):
        self.title = False
        super().__init__()

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            self.title = True

    def handle_data(self, data):
        if self.title:
            raise DataParsedException(data)

    def __enter__(self):
        return self

    def __exit__(self, t, value, traceback): # @UnusedVariable
        self.close()


class DataParsedException(Exception):
    def __init__(self, data):
        self.data = data


class DocumentRetriever(object):
    def __init__(self, filename, indexdir="index", encoding="utf-8", limitmb=5000, procs=8, mlimit=1000000000, pages=None):
        self._zim_file = ZIMFile(filename, encoding)
        self.indexdir = indexdir
        self.encoding = encoding
        self.ana = StemmingAnalyzer()
        
        if not os.path.exists(indexdir):
            if pages is not None:
                with open(pages) as pages:
                    tmp = []
                    for page in pages:
                        tmp.append(page[0:-1])
                pages = tmp

            writer_analyzer = StemmingAnalyzer(cachesize=-1)
            os.mkdir(indexdir)
            schema = Schema(idx=NUMERIC(stored=True), pidx=NUMERIC(stored=True), content=TEXT(analyzer=writer_analyzer))
            self.ix = create_in(indexdir, schema)
            writer = self.ix.writer(limitmb=limitmb / procs, procs=procs)
        
            articleCount = self._zim_file.header_fields['articleCount']
            bar = progressbar.ProgressBar(max_value=articleCount)
            for idx in range(articleCount):
                # get the Directory Entry
                data = self.get_data_by_idx(idx)
                if data is not None:
                    if pages is not None:
                        title = self.getTitle(data)

                    if pages is None or title in pages:
                        paragraphs = self.getParagraphs(data)
                        for pidx in range(len(paragraphs)):
                            writer.add_document(idx=idx, pidx=pidx, content=paragraphs[pidx])
                
                bar.update(idx)
            
            writer.commit()
        else:
            self.ix = index.open_dir(indexdir)
            
        mlimit_before = DocumentRetriever.memory_limit(mlimit)
        self.searcher = self.ix.searcher(weighting=scoring.TF_IDF())
        DocumentRetriever.memory_limit(mlimit_before)


    def get_data(self, idx, pidx):
        data = self.get_data_by_idx(idx)
        return self.getParagraphs(data)[pidx]
            
    
    def get_data_by_idx(self, idx):
        entry = self._zim_file.read_directory_entry_by_index(idx)
        if entry['namespace'] == "A":
            try:
                return self._zim_file._read_blob(entry['clusterNumber'], entry['blobNumber']).decode(self.encoding)
            except:
                return None
        else:
            return None
        
    def retrieve(self, sentence):
        parsedSentence = ' '.join([token.text for token in self.ana(sentence)])
        query = QueryParser("content", self.ix.schema).parse(parsedSentence)
        results = self.searcher.search(query, limit=1)
        try:
            return self.get_data(results[0]["idx"], results[0]["pidx"])
        except:
            query = QueryParser("content", self.ix.schema, group=qparser.OrGroup).parse(parsedSentence)
            results = self.searcher.search(query, limit=1)
            try:
                return self.get_data(results[0]["idx"], results[0]["pidx"])
            except:
                return None
            
    
    def getParagraphs(self, html):
        try:
            with DRHtmlParser() as parser:
                parser.feed(html)
                return parser.output
        except:
            return None


    def getTitle(self, html):
        try:
            parser = TitleParser()
            try:
                parser.feed(html)
            except DataParsedException as dataParsed:
                return dataParsed.data
            finally:
                try:
                    parser.close()
                except:
                    pass
        except:
            return None

        return None

    @staticmethod
    def memory_limit(mlimit):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (mlimit, hard))
        return soft

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.searcher.close()


def main():
    with DocumentRetriever(sys.argv[1], sys.argv[2], pages=sys.argv[3]) as dr:
        start = time.time()
        print(dr.retrieve("When did Albert Einstein born?"))
        print(dr.retrieve("What is a lion?"))
        print(dr.retrieve("What kind of source do you use to cook a chiken?"))
        print(dr.retrieve("Where is Obama?"))
        print(dr.retrieve("Where is South Korea?"))
        print(dr.retrieve("What makes you happy?"))
        print(dr.retrieve("What was the spell that Harry Porter used?"))
        print(dr.retrieve("Could you recommend me a role playing game?"))
        print(dr.retrieve("How can you launch successfully?"))
        print(dr.retrieve("How do you define panic disorder?"))

        print(time.time() - start)
    
if __name__ == '__main__':
    main()
