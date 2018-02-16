import utils
from datetime import date


class AgeQuestionHandler(object):

    def handle(self, resource, old_word_values):
        sentence = "{resource} {verb} {age} years old."
        birth_date, death_date = self._get_dates(old_word_values)
        if birth_date:
            return sentence.format(
                resource=utils.strip_value(resource),
                verb=self._get_verb(death_date),
                age=self._get_age(birth_date, death_date),
            )

    def _get_dates(self, old_word_values):
        birth_date, death_date = None, date.today()
        for _, prop, values in old_word_values:
            if prop == 'deathDate':
                death_date = self._string_to_date(values.pop())
            elif prop == 'birthDate':
                birth_date = self._string_to_date(values.pop())
        return birth_date, death_date

    def _string_to_date(self, date_str):
        y, m, d = map(lambda x: int(x), utils.strip_value(date_str).split('-'))
        return date(y, m, d)

    def _get_verb(self, death_date):
        return 'is' if death_date == date.today() else 'was'

    def _get_age(self, birth_date, death_date):
        return death_date.year - birth_date.year
