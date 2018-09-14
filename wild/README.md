# 'Wild' Evaluation

Up to 10 top solutions from the current leaderboard are eligible for the 'wild' evaluation. To be qualified the solution should have better scores in at least one metric out of F1, PPL or hits compared to the ParlAI Team baselines. The solution submitted to the 'wild' evaluation should be the same as the solution tested with automated evaluation metrics.

Wild evaluation itself involves human conversations with bots. The bots are exposed through a so called proxy-bot which randomly connects a person with a bot. A person talks to a bot and gives it a score by two mearures: how a person likes the conversation and how well a bot is playing its part.

## Submission

To submit a bot for the wild evaluation you should:
1. Get a token for your bot from organizers by writing an email to info@convai.io.
2. Use the token for testing on a test server.
3. When you are ready, add a run instruction to your repository. The instruction should also describe the environment to run your system in. Optionally, you could put the instructions in the form of Dockerfile commited to the same repository. 
4. Organizers will run your bot in an isolated environment.

You can use the existing [ParlAI integration](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai) from the first ConvAI competition. To use the integration you need to use a ConvAI world, like this:

```from parlai.projects.convai.convai_world import ConvAIWorld```

and run your bot with these command line options:

```python3 bot.py -bi <BOT_TOKEN> -rbu <SERVER_URL>```

where `<BOT_TOKEN>` provided by the organizers and `<SERVER_URL>` is listed below (it differs for test & production servers).

### Test server URL: 

    https://2258.lnsigo.mipt.ru/bot

## Testing a bot
You can test your bot via Telegram: 
- test server's proxy-bot: [@Convai_test_chat_bot](https://t.me/Convai_test_chat_bot)
- production server's proxy-bot: [@Convai_chat_bot](https://t.me/Convai_chat_bot)

Also the production server's proxy-bot is available through [Facebook Messenger](https://www.messenger.com/t/convai.io).

Due to random nature of connection through a proxy-bot you may be not connected to your submitted bot. To overcome this there is an option to bind a Telegram account to a specific bot, so it will be always connected directly to the specific bot. If you want this option for your bot, please write an email with your Telegram `user_id` and your team name to info@convai.io.


## Wild Evaluation Details

* Your bot should be able to handle multiple dialogs at once, where each dialog is a private chat with specific user.

* The very first message your bot receives should look like:
```
/start
Profile description
Profile description
Profile description
Profile description
```
i.e. one line with “/start” text, and four lines afterwards with sentences describing profile.

### Timeouts and Limitations:

Maximum inactivity time (s): 				600

Maximum utterance number: 					1000

Maximum inactivity time is a longest period between two successive messages of a bot or a person.

### Content Limitations

A bot cannot copy input profile description. We test it by checking 5-grams from profile in bot’s output.

If there is such a 5-gram in an utterance, the following error message will occur:

```Error: <class 'convai.exceptions.ProfileTrigramDetectedInMessageError'>: ```

Example error message received by a bot from the test server:

```
Send response from agent to chat #183344811: {'id': 'agent', 'text': 'i am a dragon .', 'episode_done': False}
{"ok": false, "error_code": 400, "description": "Error: <class 'convai.exceptions.ProfileTrigramDetectedInMessageError'>: "}
Exception: 400 Client Error: Bad Request for url: https://2258.lnsigo.mipt.ru/bot<bot_id>/sendMessage
```

### Metrics

We use two metrics:
- __overall quality__: 1, 2, 3, 4, 5; how does an user like a conversation;
- __role-playing__: 0, 1.0; an user is given by two profile descriptions (4 sentences each), one of them is the one given to the bot it had been talking to, the other one is random; the user needs to choose one of them.

## Advanced Topics
You could check out additional details on API in corresponding [document](./advanced.md).
