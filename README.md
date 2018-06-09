# ConvAI2: Overview of the competition

There are currently few datasets appropriate for training and evaluating models for non-goal-oriented dialogue systems (chatbots); and equally problematic, there is currently no standard procedure for evaluating such models beyond the classic Turing test.

The aim of our competition is therefore to establish a concrete scenario for testing chatbots that aim to engage humans, and become a standard evaluation tool in order to make such systems directly comparable.

This is the second Conversational Intelligence (ConvAI) Challenge. The previous one was conducted under the scope of NIPS 2017 Competitions track. This year we aim to improve over last year:
* providing a dataset from the beginning, Persona-Chat
* making the conversations more engaging for humans
* simpler evaluation process (automatic evaluation, followed then by human evaluation)


# Prize

The winning entry will receive $20,000 in Mechanical Turk funding -- in order to encourage further data collection for dialogue research.

# News 

- May 9: **Hackathon:** We will be organizing a *non-compulsory* hackathon around the competition: [DeepHack.Chat Hackathon](http://convai.io/#deephackchat-hackathon). The most promising team attending will receive a **travel grant** to attend NIPS 2018!!

- April 21: **Leaderboard and baselines:** Leaderboard, baseline numbers and code for training and evaluating them are up! 

# Current Leaderboard

| Model                | Creator  | PPL           | Hits@1  |   F1   |
| -------------        | ---      | :------------- | :-----  |  :----- |
|  [Seq2Seq + Attention](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines/seq2seq)  | ParlAI team          | 31.35&#x1F34E;        | 16.6       | 16.18&#x1F34E; |
|                     | Tensorborne      | 38.24   |   12.0 | 15.94   |
|  [Language Model](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines/language_model)      | ParlAI team          | 46.0         | -       |  15.02 |
|  [KV Profile Memory](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines/kvmemnn)    | ParlAI team          | -             | 55.2&#x1F34E;    |  11.9 |

&#x1F34E; denotes the current best performing model for each metric on the hidden test set.

Models by ParlAI team are baselines, and not entries into the competition; code is included for those models.

Note that the [scripts](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines) that you can run locally will give metrics on the validation set, not the hidden test set which is reported here (for that, you need to submit your code, see below). 

**Validation Leaderboard**  We also provide an additional validation set leaderboard [here](https://github.com/DeepPavlov/convai/blob/master/leaderboards.md).

# PersonaChat ConvAI2 Dataset

<img src="personachat-example.png">

Persona-Chat training set consists of conversations between crowdworkers who were randomly paired and asked to act the part of a given provided persona (randomly assigned, and created by another set of crowdworkers). The paired workers were asked to chat naturally and to get to know each other during the conversation. This produces interesting and engaging conversations that learning agents can try to mimic. 

<!-- The Persona-Chat dataset is designed to facilitate research into alleviating some of the issues that traditional chit-chat models face, and with the aim of making such models more consistent and engaging, by endowing them with a persona.
-->

The Persona-Chat task aims to model normal conversation when two interlocutors first meet, and get to know each other. Their aim is to be engaging, to learn about the other's interests, discuss their own interests and find common ground. The task is technically challenging as it involves both asking and answering questions, and maintaining a persistent persona, which is provided. 

Conversing with current chit-chat models for even a short amount of time quickly exposes their weaknesses. Common issues with chit-chat models  include:
* (i) the lack of a consistent personality [(Li et al., 2016)](https://arxiv.org/abs/1603.06155) as they are typically trained over many dialogs each with different speakers,  
* (ii) the lack of an explicit long-term memory as they are typically trained to produce an utterance given only the recent dialogue history [(Vinyals et al., 2015)](https://arxiv.org/abs/1506.05869); and  
* (iii) a tendency to produce non-specific answers like ``I don't know'' [(Li et al., 2015)](https://arxiv.org/abs/1510.03055). 

This competition aims to find models that address  those specific issues. The baseline systems we have already run indicate that there is hope we can make steps in that direction.

The dataset consists of 164,356 utterances in over 10,981 dialogs, some of which are set aside for validation. The speaker pairs each have assigned profiles coming from a set of 1155 possible personas, each consisting of at least 5 profile sentences, setting aside 200 never seen before personas for validation. To avoid modeling that takes advantage of trivial word overlap, we crowdsourced additional rewritten sets of the same personas, with related sentences that are rephrases, generalizations or specializations, rendering the task much more challenging. Evaluation will take place on both types of persona. 

More details can be found in the [paper](https://arxiv.org/abs/1801.07243) describing the dataset.

The competition dataset is available in our open source system [ParlAI](http://parl.ai), more specifically [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2).
That is, install ParlAI and then do:
```bash
python examples/display_data.py --task convai2 --datatype train
```
to look at the data.

Source code for baseline methods for the competition are also already provided in ParlAI [here](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2), including training loop and evaluation code for the automatic evaluation metrics.
Baseline results are provided in the [paper](https://arxiv.org/abs/1801.07243), although the dataset is now larger. (We have hence run new baselines that appear on the leaderboard above.)

As the original test set was released,  we have crowdsourced further data for a hidden test set  unseen by the competitors for automatic evaluation. 

# Evaluation

Competitors' models will then be compared in three ways: 
* (i) automated evaluation metrics on a new test set hidden from the competitors; 
* (ii) evaluation on Amazon Mechanical Turk; and 
* (iii) `wild' live evaluation by volunteers having conversations with the bots. 

The winning dialogue systems will be chosen based on these scores. 

## Metrics

There are three types of metrics we will evaluate:

* **Automated metrics**  - Perplexity, F1 and hits@k. These  will be computed on the hidden test set. Competitors will provide their code, and we will run the final evaluation (a validation set will be provided for their own local tests).
Perplexity is only scored for probabilistic generative models, F1 can be computed for any model that produces a response, and hits@k is computed for any model that can rank a set of candidate responses that we provide (either retrieval based models, or generative models capable of assigning a probability to a candidate response). As some methods are not applicable to some metrics, we will have a separate leaderboard for each. The top performing methods for each metric will be evaluated in the live experiments.

* **Amazon Mechanical Turk** - given the entrant's model code, we will run live experiments where Turkers chat to their model given instructions identical to the creation of the original dataset, but with new profiles, and then score its performance. Turkers will score the models between 1-5 with three metrics: fluency, consistency and engagingness. Finally the Turker will try to guess the persona being used by the bot (which was provided) as another measure of the ability of the bot to stick to its given persona. See 
[(Zhang et al., 2018)](https://arxiv.org/abs/1801.07243) for more details of these metrics and collected scores of baseline systems.

* **`Wild' Live Chat with Volunteers** - Finally, we will solicit volunteers to also chat to the models in a similar way to the Mechanical Turk setup. As volunteers, unlike Turkers, are not paid and will likely not follow the instructions as closely, the distribution will likely be different, hence serving as a test of the robustness of the models. This setup will be hosted through the Messenger and Telegram APIs.


## Protocol

We will run live volunteer chat throughout the competition so that competitors can try out their bots talking to humans and to collect live data, if they so wish (however, they are also free to only use the fixed train/test format at this stage).

The automated metrics will be used to obtain a shortlist of best performing systems, likely the top 3 scoring systems from each of the three metrics (Perplexity, F1 and hits@k). If those three leaderboards feature the same models at the top we will take  systems further down the leaderboards, up to a maximum of 10. These systems will be evaluated in the final live experiments on Mechanical Turk and via volunteers using the same scoring protocols, already described.

During NIPS the `wild' live conversation can continue, and the best performing systems will be showcased and conversed with.

We will declare winners in both the automated metrics tracks, and in the live evaluations (which will be considered the grand prize, being more important). The latter will consist of the average of the Turk and wild (volunteer) scores.
Finally, the solutions and any data collected will be made open source to the community.

# Rules

* Competitors should indicate which training sources are used to build their models, and whether (and how) ensembling is used (we may place these in separate tracks in an attempt to deemphasize the use of ensembles).
* Competitors must provide their source code so that the hidden test set evaluation and live experiments can be computed without the team's influence, and so that  the competition has further impact as those models can be released for future research to build off them. Code can be in any language but a thin python wrapper must be provided in order to work with our evaluation and live experiment code via [ParlAI's](http://parl.ai) interface.
* We will require that the winning systems also release their training code so that their work is reproducible (although we also encourage that for all systems).
* Competitors are free to augment training with other datasets as long as they are publicly released (and hence, reproducible). Hence, all entrants are expected to work on publicly available data or release the data they use to train. 

# Model Submission

To submit an entry, create a private repo with your model that works with our evaluation code, and share it with the following github accounts: emilydinan, klshuster, jaseweston, JackUrb, varvara-l, madrugado.

See [this directory](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2) for example baseline submissions.

The models should work with PyTorch 0.4, and the top level README should tell us your team name, model name, and where the eval_ppl.py, eval_hits.py etc. files are so we can run them. Those should give the numbers on the validation set. Please also include those numbers in the README so we can check we get the same. We will then run the automatic evaluations on the hidden test set and update the leaderboard. You can submit a maximum of once per month.
We will use the same submitted code for the top performing models for computing human evaluations when the submission system is locked on September 30th.


# Schedule

Up until September 30th competitors will be able to submit models (source code) to be evaluated on the hidden test set using automated metrics (which we will run on our servers). 

Ability to submit a model for evaluation by automatic metrics to be displayed on the leaderboard will be available by April 6th.
The current leaderboards will be visible to all competitors.


`Wild' live evaluation can also be performed at this time to obtain evaluation metrics and data, although those metrics will not be used for final judgement of the systems, but more for tuning systems if the competitors so wish. 


On September 30th the source code submission system will be locked, and the best performing systems will be evaluated over the next month using Mechanical Turk and the `wild' live evaluation.

Winners will be announced at NIPS 2018.


# DeepHack.Chat Hackathon

This is not a compulsory part of the competition, but you may also be interested in the following hackathon:

If you submit your solution before the **10th of June**, you can participate in the qualification round of [DeepHack.Chat](http://deephack.me/chat) hackathon which will take place in Moscow on July 2-8. We will select 10 to 12 teams whose systems score best in terms of automatic metrics, and invite them to participate in the final round of the hackathon. At the hackathon teams will further improve their systems and listen to lectures from the top researchers in the field. Participants will also take part in live human evaluation of dialogue systems of other teams. The winning team will get a travel grant to [NIPS](https://nips.cc/) to ConvAI finals.

If you want to participate in the hackathon, please include a file ``TEAM'' in your repository when you submit your models. The file should contain names and emails of the members of your team. The team should consist of no more than 5 people.

# Organizing team

The organizing team comes from multiple groups --- Moscow Institute of Physics and Technology, Facebook AI Research, University of Montreal, McGill and Carnegie Mellon University.

The Team consists of: Mikhail Burtsev, Varvara Logacheva, Valentin Malykh, Ryan Lowe, Iulian Serban, Shrimai Prabhumoye, Emily Dinan, Douwe Kiela, Alexander Miller, Kurt Shuster, Arthur Szlam, Jack Urbanek and Jason Weston.

Advisory board: Yoshua Bengio, Alan W. Black, Joelle Pineau, Alexander Rudnicky, Jason Williams.

# Partners

## Platinum Partner

<a href="https://research.fb.com/category/facebook-ai-research-fair/"><img src="2017/facebook.png"></a>


# ConvAI 2017

Webpage of the 1st NIPS Conversational Intelligence Challenge is available at [convai.io/2017](http://convai.io/2017/)
