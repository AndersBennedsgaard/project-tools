import time
import warnings
from functools import wraps

# https://wiki.python.org/moin/PythonDecoratorLibrary


def timeit(func):
    """Timer-decorator

    Args:
        func (function): Function that should be timed
    """
    @wraps(func)
    def timed(*args, **kwargs):
        ts = time.perf_counter()
        result = func(*args, **kwargs)
        te = time.perf_counter()
        print(f"\tRan {func.__name__!r} in {te-ts:.2f} seconds")
        return result
    return timed


def retry(tries, delay=3.0, backoff=2.0):
    """Retries a function or method, until it returns True

    Args:
        tries (int): Number of retries.
        delay (float, optional): Delay between each retry in seconds. Defaults to 3.0.
        backoff (float, optional): Delay is multiplied with this after each retry, creating exponentially increasing
        waiting times. Defaults to 2.0.
    """
    if backoff < 1:
        raise ValueError("backoff must be one or greater")

    tries = int(tries // 1)
    if tries < 0:
        raise ValueError("tries must be 0 or greater")

    if delay <= 0:
        raise ValueError("delay must be greater than 0")

    def deco_retry(func):
        @wraps(func)
        def func_retry(*args, **kwargs):
            mtries, mdelay = tries, delay  # make mutable

            rv = func(*args, **kwargs)
            while mtries > 0:
                if rv is True:
                    return True

                mtries -= 1
                time.sleep(mdelay)
                mdelay *= backoff

                rv = func(*args, **kwargs)

            return False  # Ran out of tries
        return func_retry  # true decorator -> decorated function
    return deco_retry  # @retry(arg[, ...]) -> true decorator


class countcalls:
    """Decorator that keeps track of the number of times a function is called.

    Example:

    @countcalls\n
    def f():
    print('f called')

    f()\n
    f()\n
    print f.count() # prints 2\n
    print countcalls.counts() # same as f.counts()
    """

    __instances = {}

    def __init__(self, f):
        self.__f = f
        self.__numcalls = 0
        countcalls.__instances[f] = self

    def __call__(self, *args, **kwargs):
        self.__numcalls += 1
        return self.__f(*args, **kwargs)

    def count(self):
        """Return the number of times the function f was called.

        Returns:
            int: Number of times the function/method has been called
        """
        return countcalls.__instances[self.__f].__numcalls

    @staticmethod
    def counts():
        """Return a dict of {function: # of calls} for all registered functions.

        Returns:
            dict: A count for each function that has been called
        """
        return dict([(f.__name__, countcalls.__instances[f].__numcalls) for f in countcalls.__instances])


def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func
