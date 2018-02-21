import sys


# http://ozzmaker.com/add-colour-to-text-in-python/
colors_ = ['black', 'red', 'green', 'yellow',
           'blue', 'purple', 'cyan', 'white']
fg_colors = dict([(c, i + 30) for i, c in enumerate(colors_)])
bg_colors = dict([(c, i + 40) for i, c in enumerate(colors_)])
# XXX What about 4?
styles = {'none': 0, 'bold': 1, 'uline': 2, 'neg1': 3, 'neg2': 5}


def code(style='none', fg=None, bg=None):
    attrs = [styles[style]]
    if not fg is None:
        attrs.append(fg_colors[fg])
    if not bg is None:
        attrs.append(bg_colors[bg])
    return u'\033[' + ';'.join([str(a) for a in attrs]) + 'm'


def reset_code():
    return code()


def colorize(s, **kwargs):
    return code(**kwargs) + s + reset_code()
