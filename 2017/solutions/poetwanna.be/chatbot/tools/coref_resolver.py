from corenlp import CoreNLPWrapper


class CoreferenceResolver(object):

    def __init__(self, nlp_server_url, max_hist=20):
        self.max_hist = max_hist
        self.nlp = CoreNLPWrapper(nlp_server_url)
        self.ignore = set("i me my mine myself".split())
        self.ignore |= set("you your yours yourself yourselves".split())
        self.ignore |= set("we us our ours ourself ourselves".split())
        self.ignore |= set(
            "herself himself themself itself themselves".split())

        self.accepted_tags = ["PRP", "PRP$"]

    def new_state(self):
        return []

    def _update_state(self, state, text):
        #text = self.nlp.clean_text(text)
        tokens_no = len(self.nlp.tokenize(text)["tokens"])
        state.append((text, tokens_no))
        if len(state) >= self.max_hist:
            state = state[1:]

        return state

    def _resolve(self, state):
        nlp_out = self.nlp.coref(" ".join([utt for (utt, _) in state]))
        corefs = nlp_out["corefs"]
        sents = [s["tokens"] for s in nlp_out["sentences"]]

        for key in corefs:
            subs = corefs[key]
            sent_no = subs[0]["sentNum"] - 1
            start_idx = subs[0]["startIndex"] - 1
            end_idx = subs[0]["endIndex"] - 1
            resolution = self._collapse(sents[sent_no][start_idx:end_idx])
            pos_tag = sents[sent_no][start_idx]["pos"]

            if end_idx - start_idx == 1 and pos_tag in self.accepted_tags:
                continue

            if len(subs) <= 1 or resolution.lower() in self.ignore:
                continue

            for sub in subs[1:]:
                sent_no = sub["sentNum"] - 1
                start_idx = sub["startIndex"] - 1
                end_idx = sub["endIndex"] - 1
                pos_tag = sents[sent_no][start_idx]["pos"]

                # We only substitute one-word phrases that have appropriate POS
                # tags
                if end_idx - start_idx == 1 and pos_tag in self.accepted_tags:
                    word = self._collapse(sents[sent_no][start_idx:end_idx])
                    if pos_tag == "PRP$" and word.lower() != "her":
                        add = "'s"
                    else:
                        add = ""
                    if word not in self.ignore:
                        sents[sent_no][start_idx][
                            "originalText"] = resolution + add
                        end_offset = sents[sent_no][
                            end_idx-1]["characterOffsetEnd"]
                        sents[sent_no][start_idx][
                            "characterOffsetEnd"] = end_offset

        resolved_state = [tkn for snt in sents for tkn in snt]

        return resolved_state

    def _collapse(self, tokens):
        output = ""
        prev_end = 0
        for token in tokens:
            begin = token["characterOffsetBegin"]
            if begin < prev_end:
                continue

            if prev_end < begin:
                output += " "

            output += token["originalText"]
            prev_end = token["characterOffsetEnd"]

        return output.strip()

    def set_article(self, state, text):
        state = self._update_state(state, text)
        resolved_state = self._resolve(state)
        tokens_no = len(resolved_state)

        resolved_article = self._collapse(
            resolved_state[tokens_no-state[-1][1]:])

        return state, resolved_article

    def bot_utterance(self, state, text):
        state = self._update_state(state, text)
        return state

    def user_utterance(self, state, text):
        state, self._update_state(state, text)
        resolved_state = self._resolve(state)
        tokens_no = len(resolved_state)

        resolved_user_utt = self._collapse(
            resolved_state[tokens_no-state[-1][1]:])

        return state, resolved_user_utt
