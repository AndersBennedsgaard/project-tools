import time
from functools import wraps


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
