# Advanced Topics
## Bot API Documentation

Bots are connected to the evaluation system via simplified version of [Telegram Bot API](https://core.telegram.org/bots/api). There are only two methods are available: `getUpdates` and `sendMessage`. Only text messages are allowed. The method `sendMessage` ignores all other fields besides `chat_id` and `text` (and additionally out unique field `msg_evaluation` described below).  The method `getUpdates` allows to use only `limit` and `timeout` fields. The system provides updates only of type `message`.

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
