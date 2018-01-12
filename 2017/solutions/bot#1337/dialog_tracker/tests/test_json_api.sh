#!/bin/bash

CHAT_ID="ab2c4318-d63e-405a-89e2-bfcc6880acdb"

curl -H "Content-Type: application/json" -XPOST \
  -d '{"text": "The name-letter effect is the tendency of people to prefer the letters in their name over other letters in the alphabet. Discovered in 1985 by the Belgian psychologist Jozef Nuttin, the effect has been replicated in dozens of studies."}' \
  http://0.0.0.0:5000/start

echo {"\"chat_id\"": "\"$CHAT_ID\"", "\"text\"": "\"Hi!\""} | \
curl -H "Content-Type: application/json" -XPOST -d @- \
  http://0.0.0.0:5000/message

# curl -H "Content-Type: application/json" -XPOST \
#   -d '{"chat_id": '$CHAT_ID', "text": "How are you?"}' \
#   http://0.0.0.0:5000/message

# curl -H "Content-Type: application/json" -XPOST \
#   -d '{"chat_id": '$CHAT_ID', "text": "What is your name?"}' \
#   http://0.0.0.0:5000/message

# curl -H "Content-Type: application/json" -XPOST \
#   -d '{"chat_id": '$CHAT_ID', "text": "Ask me question"}' \
#   http://0.0.0.0:5000/message

# curl -H "Content-Type: application/json" -XPOST \
#   -d '{"chat_id": '$CHAT_ID', "text": "I have to go. Bye!"}' \
#   http://0.0.0.0:5000/message

# curl -H "Content-Type: application/json" -XPOST \
#   -d '{"chat_id": '$CHAT_ID'}' \
#   http://0.0.0.0:5000/end
