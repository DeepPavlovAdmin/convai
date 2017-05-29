# Participants’ Bots API Description

We’re using trimmed version of Telegram Bot API to communicate with participants’ bots. The bot should be able to connect to custom Telegram server running on our host with provided token. Documentation on API is available [here](https://core.telegram.org/bots/api).

## Updates

You’re able to use webhooks or `getUpdates` method to [get new updates](https://core.telegram.org/bots/api#getting-updates). We’ll send you only updates of [type “message”](https://core.telegram.org/bots/api#update). You’re able to request updates of other types but you’ll never receive them.

## Messages

Your bot should be ready to do several chats “simultaneously”. Every single chat with another bot or a human is presented as chat with a new user. This [user](https://core.telegram.org/bots/api#user) will have unique `ID = chat_id` and first name “Anonym”. The very first message of every chat contains classic `/start` command and context message. For example: `/start You're sitting watching TV, and suddenly you discover a wasp crawling on your wrist. What you gonna do?` The following messages from the server will contain only the text written by human or another bot. `/begin` command means that the bot should be the one who’s replying to the context message. `/end` command means the end of the chat. If the bot doesn’t reply during 5 minutes we’re considering the chat as finished and sending `/end` command to the bot. The bot is able to stop chat by itself by sending message with text `/end`. In case of some errors the bot will receive `/end` command with error description. E.g.: `/end Invalid JSON`.

The bot’s replies are JSON-encoded. E.g.:

```json
{
    "text": "Hello",
    "evaluation": 10
}
```

Evaluation field is optional for reply to the context message.

We’re suggesting you to test your bot using Telegram messenger client. Setup your bot to work with Telegram server: get the token from [@BotFather](https://core.telegram.org/bots#botfather) and specify `https://api.telegram.org` API host in your config (you should have such parameter, not a hardcoded value). Typical chat with the bot should look like this (the tester is sending messages as “Server” here using her messenger):

**Server:** /start You're sitting watching TV, and suddenly you discover a wasp crawling on your wrist. What you gonna do?

**Bot:**

```json
{
    "text": "What’s a wasp?"
}
```

**Server:** A stinging bug that flies.

**Bot:**

```json
{
    "text": "Oh, how strange. Do they still exist? I’ve never seen one",
    "evaluation": 10
}
```

**Server:** They died out because of the dust.

**Bot:**

```json
{
    "text": "Oh, sorry, I’ve forgotten about a pie in an oven completly. Bye!",
    "evaluation": 0
}
```

**Server:** OK. See you!

**Bot:**

```json
{
    "text": "/end",
    "evaluation": {
        "quality": 7,
        "breadth": 9,
        "engagement": 5
    }
}
```

### Evaluation

Every bot’s reply should contain `evaluation` field. This is the bot’s quality measurement of its partner's replies: integer number from 1 to 10. `1` means “very bad reply” and `10` is a “very good reply”.

The last message of every chat should have text `/end` and three evaluation fields measuring quality, breadth and engagement of the chat (integer, 1 to 10) like this:

```json
{
    "text": "/end",
    "evaluation": {
        "quality": 3,
        "breadth": 5,
        "engagement": 2
    }
}
```

## Available API methods

The only supported Telegram Bot API methods are `getUpdates` and [`sendMessage`](https://core.telegram.org/bots/api#sendmessage).
