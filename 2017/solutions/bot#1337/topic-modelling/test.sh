#!/bin/bash

curl -H "Content-Type: application/json" -XPOST \
  -d '{"text": "ParlAI, released this year, is a unified platform for training and evaluating AI models on a variety of openly available dialog datasets using open-sourced learning agents. Applicants for the grants will be expected to either contribute to the pool of available agents, e.g. by research into new strongly performing models and/or add to the pool of available tasks that are useful for training and evaluating those agents."}' \
  http://0.0.0.0:3000/respond
