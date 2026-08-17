import time
def time_call(func, *args):
    start=time.perf_counter()
    result=func(*args)
    end=time.perf_counter()
    duration=end-start
    return (result,duration)
