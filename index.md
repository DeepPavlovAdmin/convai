# Participate

Register as a team member | Register as a human evaluator
:---: | :---:
link | link


# Timeline

Date | Milestone
---: | ---
_April, 2017_ | **Registration is open.** Registered teams are allowed to submit solutions for the Qualification Round. 
_18th of June, 2017_ | **Qualification Round is closed.** Qualification Leaderboard is published. Teams for the Human Evaluation Round are selected.
_24th-30th of July, 2017_ | **Human Evaluation Round.** One week NLP summer school. Teams, school participants and volunteers evaluate entries of qualified teams.
_1st of September, 2017_ | **1st Dataset is published.** Data collected at the Human Evaluation Round is published. Teams tune their solutions on the 1st Dataset.
_12th of November, 2017_ | **Submission of conversational agents is closed.** Teams submit final solutions for the NIPS Live Competition.
_20th of November - 3rd of December, 2017_ | **Pre-NIPS Human Evaluation.** Teams and volunteers start to evaluate entries of teams.
_4th-9th of December, 2017_ | **Conversational Intelligence Live Competition at NIPS.** Teams, conference participants and volunteers continue to evaluate entries of teams. Announcement of winners. 

# Overview of the Competition

Today, evaluation of dialogue agents is severely limited by the absence of accurate formal metrics. Existing statistical measures such as perplexity, BLEU, recall and others are not sufficiently correlated with human evaluation [1]. Blind assessment of communication quality by humans is a straightforward solution famously proposed by Alan Turing as a test for machine intelligence [2]. Unfortunately, human assessment is time and resource consuming. Here we propose to crowdsource evaluation of dialogue systems in the form of a live competition. Participants of the competition, as well as volunteers, will be asked to perform a blind evaluation of a discussion about a news/wikipedia article with either a bot or a human peer. As a result we expect to have two outcomes: (1) a measure of quality of state of the art dialogue systems compared to human level, and (2) an open source dataset collected from evaluated dialogs.

[1] Liu, Chia-Wei, et al. "How NOT to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation." arXiv preprint arXiv:1603.08023 (2016).

[2] Turing, Alan M. "Computing machinery and intelligence." Mind 59.236 (1950): 433-460.

# Competition Rules

## Competition rounds

The competition consists of four rounds.

1. **Qualification round.** Registered participants submit their results on the task **!--(to be selected)--!**. Submission of results is closed on the **_18th of June, 2017_**. All teams with working submissions are selected for the Human evaluation round.
2. **Human Evaluation Round.** Members of selected teams are invited to participate in a week long NLP summer school by giving a talk on their research. Participation can be on site or remote. During the school week members of teams, school participants, and volunteers recruited via the competition web page evaluate the submitted dialogue systems on the competition task. At the end of Human Evaluation Round, up to 10 teams are selected for the NIPS Live Competition. Every team is required to evaluate at least 150 dialogs during the Round.
3. **Tuning round.** Dataset of rated dialogs collected during the Human Evaluation Round is open sourced and can be used by participating teams to tune their solutions.
4. **NIPS round.** Starting two weeks before the NIPS conference teams and volunteers perform evaluation of submitted dialog systems. At the beginning of NIPS the conference participants are invited to volunteer in evaluation of teams’ entries adjusted over the Tuning Round. Final rating of submissions is presented on the Competition session at NIPS.

## Task

Both human evaluators and dialogue agents complete the same task.

1. Connect randomly with a peer. The peer might be a chat bot or other human user. No information about identity of the peer is provided.
2. Both parties are given a text of a recent news/wikipedia article.
3. Discuss content of the article with the peer as long as you wish.
4. Choose another news/wikipedia article and/or anonymous peer.

## Evaluation

1. Evaluator will not be given any information about identity of the peer.
2. Members of the team will be automatically excluded from evaluation of their own submission and each other.
3. The quality of every response is subjectively evaluated on the 0 to 10 range.
4. The quality of the dialog as a whole as well as its breadth and engagement are evaluated on the 0 to 10 range.
5. Final rating is calculated as an average of evaluation values accumulated by submission during the NIPS Round of Competition.

## Technical infrastructure

1. Competitors will provide their solutions in the form of executable source code supporting a common interface (API).
2. These solutions will be run in isolated virtual environments (containers).
3. The solutions will not be able to access any external services or the Internet, and will only be able to communicate with the supervisor bot to guard against cheating.
4. The master bot will facilitate communication between human evaluators and the competitors’ solutions. It will be available in popular messenger services (Facebook/Telegram). It’s main function will be to connect a participant to a (randomly selected) solution or peer and log the evaluation process.
5. The master bot will provide the instructions and a context necessary for human evaluation of presented solutions.

## Dataset

Dataset collected during competition will be distributed under open source license.

# Organizers

Mikhail Burtsev, Valentin Malykh, _MIPT, Moscow_

Ryan Lowe, _McGill University, Montreal_

Iulian Serban, Yoshua Bengio, _University of Montreal, Montreal_

Alexander Rudnicky, Alan W. Black,  Carnegie Mellon University, Pittsburgh
