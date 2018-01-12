#!/usr/bin/env bash
cd /opt/corenlp/src
exec java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
