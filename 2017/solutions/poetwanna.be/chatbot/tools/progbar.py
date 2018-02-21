from __future__ import print_function

import datetime as dt
import sys


def file_len(fname):
    import subprocess
    p = subprocess.Popen(['wc', '-l', fname],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


class Timer(object):

    def __init__(self, msg=None):
        self._start = dt.datetime.now()
        self._end = None
        self.msg = msg
        if msg is not None:
            sys.stdout.write('{: <59}'.format(msg))
            sys.stdout.flush()

    def __str__(self):
        if self._end is None:
            diff = dt.datetime.now() - self._start
        else:
            diff = self._end - self._start
        diff = diff - dt.timedelta(microseconds=diff.microseconds)
        return '[%ss]' % diff

    def tok(self, end_msg=None):
        self._end = dt.datetime.now()
        if self.msg is None and end_msg is None:
            return
        m = ' DONE ' + str(self)
        if end_msg is not None:
            m += '\n    ' + end_msg
        print(m)


class Progbar(object):

    def __init__(self, num_iters, check_every=1000):
        self.t0 = dt.datetime.now()
        self.t_last = self.t0
        self.num_iters = num_iters
        self.check_every = check_every

    def print_progress(self, this_iter, text=''):
        if this_iter == 0 or this_iter % self.check_every:
            return
        t_left = (dt.datetime.now() - self.t0) / this_iter * \
                 (self.num_iters - this_iter)
        t_left = dt.timedelta(seconds=int(t_left.total_seconds()))
        self.t_last = dt.datetime.now()
        print('\033[K\rProcessed %.2f%% (ETA: %s) %s' %
              (100.0 * this_iter / self.num_iters, t_left, text),
              end='')


class FileProgbar(Progbar):
    '''Progress monitoring for processing a file line by line'''

    def __init__(self, fpath, num_lines=None, **kwargs):
        if num_lines is None:
            print('Counting lines... ', end='')
            num_lines = file_len(fpath)
            print(num_lines)
        super(FileProgbar, self).__init__(num_lines, **kwargs)
