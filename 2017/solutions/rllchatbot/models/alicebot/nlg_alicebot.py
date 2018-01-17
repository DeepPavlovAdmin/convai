"""
Natural language generation module ALICEbot.
"""

import aiml

class NLGAlice():

    def __init__(self):

        # Load ALICE kernel.
        # The default brainpath should be "alicekernel_*.brain",
        # where '*' indicates the most recent updated date.
        self.kernel = aiml.Kernel()
        # TODO: Currently We assume that this brainpath should be in the config.
        #self.kernel.loadBrain('./models/response_models/alicebot/alicekernel_20170321.brain')
        #self.kernel.loadBrain('./models/response_models/alicebot/alicekernel_20170617.brain')
        #self.kernel.loadBrain('./models/response_models/alicebot/alicekernel_20170629.brain')
        #self.kernel.loadBrain('./models/response_models/alicebot/alicekernel_20170701.brain')
        #self.kernel.loadBrain('./models/response_models/alicebot/alicekernel_20170709.brain')
        self.kernel.loadBrain('/root/convai/models/alicebot/alicekernel_20170711.brain')

    def compute_responses(self, dialogue_history, nlu_rep, user=None):
        # TODO: Currently we assume that nlu_rep is a dictionary and it contains
        # the raw_input. [:-5] to remove the ' </s>' tag which cannot be recognized
        # by alice bot.

        if len(dialogue_history) > 5:
            dialogue_history_shortened = dialogue_history[-5:]
        else:
            dialogue_history_shortened = dialogue_history

        dialogue_history_flattened = ' '.join(dialogue_history_shortened).replace(".", "").replace("!", "").replace("?", "").replace(",", "").lower()

        #print 'dialog hist:', dialogue_history
        #print 'session:', self.kernel._sessions['_global']

        input_= dialogue_history
        for i in range(len(input_)):
            if len(input_[i]) > 500:
                input_[i] = input_[i][:500]

        # Generate response. Repeat generation step if it contains profanity,
        # if it is a repetition or if it's repeating a long subphrase.
        generation_attempt_n = 0
        while (generation_attempt_n < 3):
            generation_attempt_n += 1
            response_kernel = self.kernel.respond(input_).strip()

            response_kernel_no_backslash = response_kernel.replace("\\", "")
            if response_kernel_no_backslash.count("\"") == 2:
                response_kernel_no_backslask_split = response_kernel_no_backslash.split("\"")
                if len(response_kernel_no_backslask_split) == 3:
                    if response_kernel_no_backslask_split[1] > 3:
                        response_kernel = ""
                        continue

            break

        confidence_score = 0.5

        # If incompleteness_flag triggered, score should be low
        incompleteness = self.kernel.get_incompleteness()
        if incompleteness:
           confidence_score = 0.0

        # Handle cases when the response is incomplete like "I am a ,"
        for idx in range(len(response_kernel)):
            if response_kernel[idx] in [',', '.', '?', '!']:
                if response_kernel[idx-1] == ' ':
                    confidence_score = 0.0

        # Handle incomplete responses
        if response_kernel.replace(" ", "").replace(".", "").replace("!", "").replace("?", "").replace("'", "").replace(",", "") == "Imaandyouarea":
            response_kernel = ""
            confidence_score = 0.0

        # Remove backslashes and quotation marks
        response_kernel = response_kernel.replace("\"", "").replace("\\", "")

        priority = self.kernel.get_priority()
        confidence_score = 1.0 if priority else confidence_score

        # Enable priority for responses involving "favorite" elements (e.g. "what is your favorite movie?")
        if "favorite" in dialogue_history[-1] and "favorite" in response_kernel:
            priority = True
            confidence_score = 1.0

        return response_kernel

if __name__ == "__main__":
    # Start chat loop
    alicebot = NLGAlice()

    while True:
        user_utterance = raw_input("USER: ")

        if user_utterance == "bye":
            break

        system_utterance = alicebot.compute_responses([user_utterance], None)

        print "Bot: {}".format(system_utterance)
