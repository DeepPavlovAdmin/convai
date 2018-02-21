import glob
import pdb
import re
import numpy as np

aiml_file_list = glob.glob("./*.aiml")
bot_attributes = []
p = re.compile("<([ a-z=\"]*)\"\/>")

for f_name in aiml_file_list:
    with open(f_name) as f:
        lines = f.readlines()
        for ln in lines:
            if '<bot name=' in ln:
               cand = list(set(p.findall(ln)))
               print cand
               #for cd in cand:
               #    if cd not in bot_attributes:
               #        bot_attributes.append(cd)
               #        print "New attrbiute added: " + cd + "\n" 
    print "File: " + f_name + " is finished..\n"
print "Completed."
np.save('bot_attributes.pkl', bot_attributes)

pdb.set_trace()
