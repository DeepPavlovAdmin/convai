# Store the history of logs into the database
# Should store the logs in the same database as the server is being stored

import pymongo

MONGO_PORT = 8091
client = pymongo.MongoClient("132.206.3.23", MONGO_PORT)
db = client.convai
dialogs = db.local

# stores the log history in the database by dialog id


def store_data(dialog_id, dialog_history):
    dialogs.insert({'dialogId': dialog_id, 'logs': dialog_history })

# Stub: will use it later
def match_data():
    db_logs = list(dialogs.find({"dialogId": dialog_id}))
    if len(db_logs) > 0:
        thread = db_logs[0]['thread']
        new_thread = []
        for i, dl in enumerate(thread):
            dh = dialog_history[i]
            dl['sender'] = dh['sender']
            if dl['sender'] == 'bot':
                dl['policyID'] = dh['policyID']
                dl['model_name'] = dl['model_name']
            new_thread.append(dl)
        dialogs.update({"dialogId": dialog_id}, {'$set': {'thread': new_thread}})
