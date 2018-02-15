#
# Jan Chorowski 2017, UWr
#
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import cStringIO as StringIO
import sys
import traceback
import requests
import bs4


def to_utf8(key):
    if isinstance(key, unicode):
        return key.encode('utf8')
    return key


def U(string):
    if isinstance(string, str):
        try:
            string = unicode(string, 'utf8')
        except:
            try:
                string = unicode(string)
            except:
                pass
    return string


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called.

    After: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                _Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


class _SingletonWithPath(type):
    """ A metaclass that creates a Singleton base class when called.

    After: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = defaultdict(dict)

    def __call__(cls, path):
        instances = cls._instances[cls]
        if path not in instances:
            instances[path] = super(_SingletonWithPath, cls).__call__(path)
        return instances[path]


class SingletonWithPath(
        _SingletonWithPath('SingletonMetaWithPath', (object,), {})):
    pass


def get_ex_info():
    _ex_type, ex, tb = sys.exc_info()
    memfile = StringIO.StringIO()
    try:
        memfile.write(ex.__class__.__name__ + ' "')
        memfile.write(ex)
    except:
        pass
    memfile.write('\nTraceback (most recent call last):\n')
    traceback.print_tb(tb, file=memfile)
    err_str = memfile.getvalue()
    memfile.close()
    return err_str


# https://stackoverflow.com/a/377028
def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def list_talkers(mod=None):
    import talker
    if mod is None:
        mod = talker
    for __n in dir(mod):
        __t = getattr(mod, __n.split('.')[-1])
        __is_talker = False
        try:
            if issubclass(__t, talker.base.ResponderRole):
                __is_talker = True
        except:
            pass
        if __is_talker:
            yield __t


def fetch_random_wikinews(approx_len=500):
    rr = requests.get("https://en.wikinews.org/wiki/Special:Random")
    soup = bs4.BeautifulSoup(rr.content, 'lxml')
    title = soup.find("h1", {"id": "firstHeading"})
    content = soup.find("div", {"id": "mw-content-text"})
    paras = [title.text]
    tot_len = len(paras[-1])
    for p in content.find_all("p"):
        try:
            if "published" in list(p.children)[0]["class"]:
                paras.append('')
                continue
        except:
            pass
        if tot_len > 500:
            break
        paras.append(p.text)
        tot_len += len(paras[-1])
    paras.append(u"source: " + rr.url)
    return u'\n'.join(paras)
