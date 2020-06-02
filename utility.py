import time

# measure method execution time
def timer(func): 
    def wrapper(*args, **kwargs): 
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        print('Total took {} seconds'.format(end_time - start_time))
        return func_return
    return wrapper