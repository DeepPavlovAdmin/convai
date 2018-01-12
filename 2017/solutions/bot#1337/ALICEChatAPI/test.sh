#!/bin/bash

curl -H "Content-Type: application/json" -XPOST \
  -d '{"sentences": ["Are you bot?"]}' \
  http://0.0.0.0:3000/respond


curl -H "Content-Type: application/json" -XPOST \
  -d '{"sentences": ["Hi!", "How are you?", "Do you know the text?"]}' \
  http://0.0.0.0:3000/respond
