# 'Wild' Evaluation

Up to 10 top solutions from current leaderboard are eligible for the 'wild' evaluation. To be qualified the solution should have better scores in at least one metric out of F1, PPL or hits compared to ParlAI baselines. Solution submitted to the 'wild' evaluation should be the same as solution tested with automated evaluation metrics.

## Submission

To submit bot for the wild evaluation you should:
1. get the token for your bot from organizers by writing an email to info@convai.io ;
2. use the token for testing on test server;
3. when you are ready, send your system in form of Docker container or detailed run instruction to the organizers;
4. Organizers run your bot in isolated environment.

You can use an existing ParlAI integration from the first ConvAI competition [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai).

## Bot API Documentation

Bots are connectesd to the evaluation system via simplified version of [Telegram Bot API](https://core.telegram.org/bots/api). There are only two methods are available: `getUpdates` and `sendMessage`. Only text messages are allowed. The method `sendMessage` ignores all other fields besides `chat_id` and `text` (and additionally out unique field `msg_evaluation` described below).  The method `getUpdates` allows to use only `limit` and `timeout` fields. The system provides updates only of type `message`.

A bot should be able to handle multiple dialogs at once, where each dialog is a private chat with specific user.

### Test server URL: 

    https://2258.lnsigo.mipt.ru/

### Production server URL:

    https://2242.lnsigo.mipt.ru/ 

## Server API usage:

### Method getUpdates:

`curl https://2258.lnsigo.mipt.ru/bot<bot_token>/getUpdates`

Reply, a new chat is started:
```json
{  
   "ok":true,
   "result":[  
      {  
         "update_id":88,
         "message":{  
            "message_id":0,
            "from":{  
               "id":954017548,
               "is_bot":true,
               "first_name":"0"
            },
            "chat":{  
               "id":954017548,
               "first_name":"0",
               "type":"private"
            },
            "date":1530526222,
            "text":"/start\nmy cats are very special to me.\ni'm a construction worker.\ni enjoy building houses.\ni have 5 cats.\nmy dad taught me everything i know."
         }
      }
   ]
}
```

The very first message should look like:
```
/start
Profile description
Profile description
Profile description
Profile description
```
i.e. one line with “/start” text, and four lines afterwards with sentences describing profile.

### Method sendMessage:

`curl --request POST --header "Content-Type: application/json" --data '{"chat_id":327392578,"text":"{\"text\":\"Hello!\"}"}' https://2258.lnsigo.mipt.ru/bot<bot_token>/sendMessage`

Optionally, a message could include field `msg_evaluation` with evaluation of counterpart’s message. This field should contain 0 or 1 if you are evaluating the previous counterpart’s message. Also it could contain object with fields `message_id` and `score`, in that case the evaluation will be attached to message with message_id. Example:

Evaluation of the last message:
```json
{
  "text": "message text",
  "msg_evaluation": 1
}
```

Evaluation of a specific message:
```json
{
  "text": "message text",
  "msg_evaluation": {"score": 0, "message_id": 12}
}
```

### Timeouts and limitations:

Maximum inactivity time (s): 				600

Maximum utterance number: 					1000

Maximum number of invalid messages sent by a bot:			10

## Additional Limitations

A bot cannot copy input profile description. We test it by checking 5-grams from profile in bot’s output.

If there is such a 5-gram in an utterance, the following error message will occur:

```
Send response from agent to chat #183344811: {'id': 'agent', 'text': 'i am a dragon .', 'episode_done': False}
{"ok": false, "error_code": 400, "description": "Error: <class 'convai.exceptions.ProfileTrigramDetectedInMessageError'>: "}
Exception: 400 Client Error: Bad Request for url: https://2258.lnsigo.mipt.ru/bot<bot_id>/sendMessage
```

## Metrics

We use two metrics:
- __overall quality__: 1, 2, 3, 4, 5; how does an user like the conversation ;
- __role-playing__: 0, 1.0; an user is given by two profile descriptions (4 sentences each), one of them is the one given to the bot it had been talking to, the other one is random; the user needs to choose one of them.
