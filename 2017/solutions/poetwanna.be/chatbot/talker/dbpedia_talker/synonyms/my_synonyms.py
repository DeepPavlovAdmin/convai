class MySynonyms(object):

    MY_SYNONYMS = [
        ['category', 'hypernym'],
        ['movie', 'movies', 'starring', 'playing'],
        ['wife', 'spouse'],
        ['club', 'clubs', 'teams', 'team']
    ]
    CONDITIONAL_SYNONYMS = {
        'where': [
            ['die', 'died', 'death', 'deathplace'],
            ['born', 'birth', 'birthplace'],
        ],
        'when': [
            ['die', 'died', 'death', 'deathdate'],
            ['born', 'birth', 'birthdate'],
        ],
        'how': [
            ['old', 'older', 'birthdate', 'deathdate'],
            ['die', 'died', 'death', 'deathcause'],
        ]
    }

    def find_synonyms(self, word, tokens):  # + word stemm?
        synonyms = []
        for synonyms_list in self.MY_SYNONYMS:
            if word in synonyms_list:
                synonyms += synonyms_list
        for condition, synonyms_lists in self.CONDITIONAL_SYNONYMS.iteritems():
            if condition not in map(lambda s: s.lower(), tokens):
                continue
            for synonyms_list in synonyms_lists:
                if word in synonyms_list:
                    synonyms += synonyms_list
        return synonyms
