import aiml
import glob
import pdb

def test_savebrain():
    folder = glob.glob("./aiml-en-us-foundation-alice/*.aiml")
    k = aiml.Kernel()
    for file in folder:
       k.learn(file)
    k.saveBrain('alicekernel_20170711.brain')


def test_profile_answers():
    profile_question_list = [
    #"what is your name?",
    #"who is your father?",
    #"where are you from?",
    "how old are you?"]

    k = aiml.Kernel()
    k.loadBrain("alicekernel_20170701.brain")
    k.respond(profile_question_list[0])
    k.get_priority()

if __name__ == "__main__":
   test_savebrain()
   #test_profile_answers()
