#!/usr/bin/env python

import aiml
import os
import sys
import subprocess


class Chatbot():
  def __init__(self):
    self._kernel = aiml.Kernel()

  def initialize(self, aiml_dir):
    self._kernel.learn(os.sep.join([aiml_dir, '*.aiml']))
    properties_file = open(os.sep.join([aiml_dir, 'bot.properties']))
    for line in properties_file:
      parts = line.split('=')
      key = parts[0]
      value = parts[1]
      self._kernel.setBotPredicate(key, value)

  def respond(self, input, session_id):
    response = self._kernel.respond(input, sessionID=session_id)
    return response

  def reset(self):
    self._kernel.resetBrain()


def main():
  chatbot = Chatbot()
  chatbot.initialize("aiml-dir") # parameter is the aiml directory

  while True:
    n = input("Input: ")
    print(chatbot.respond(n))

if __name__ == '__main__':
  main()
