#
# Jan Chorowski 2017, UWr
#
'''
Wrappers for talkers that use Celery to work asynchronously
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import talker  # pylint: disable=unused-import
import talker.base

from async.celery import app


class CeleryTalker(object):
    def __init__(self, talker_class):
        super(CeleryTalker, self).__init__()
        self.name = 'async_' + talker_class.name
        base_name = 'async.%s.%s.' % (
            talker_class.__module__, talker_class.__name__)
        self.talker_class = talker_class
        self.talker = None

        for fun in ['new_state',
                    'respond_to',
                    '_respond_to',
                    'set_article',
                    'follow_up',
                    ]:
            def wrap_wrap_f(fun=fun):
                @app.task(name=base_name + fun)
                def wrap_f(*args, **kwargs):
                    t = self.talker
                    if t is None:
                        t = self.talker = self.talker_class(async=True)
                    f = getattr(t, fun)
                    return f(*args, **kwargs)
                return wrap_f
            setattr(self, fun, wrap_wrap_f())


for __n in dir(talker):
    __t = getattr(talker, __n)
    __is_talker = False
    try:
        if issubclass(__t, talker.base.ResponderRole):
            __is_talker = True
    except:
        pass
    if not __is_talker:
        continue
    __wt = CeleryTalker(__t)

    def __wrap_tf(__wt=__wt):
        def __tf(async):
            assert async
            return __wt
        return __tf
    globals()[__n] = __wrap_tf()
