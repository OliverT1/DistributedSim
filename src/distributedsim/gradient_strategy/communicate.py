import torch.distributed as dist
import inspect
import datetime

def mps_compatible(func):
    def wrapper(tensor, *args, **kwargs):
        # prev_call = inspect.stack()[1]
        # print(f'{datetime.datetime.now()} calling function {func.__name__} from location {prev_call.filename}:{prev_call.lineno} {prev_call.function}')
        if tensor.device.type == 'mps':
            tmp = tensor.data.to('cpu')
            # Call the function on CPU
            result = func(tmp, *args, **kwargs)
            # Copy the result back to mps if needed
            tensor.data.copy_(tmp.to('mps'))
            return result
        else:
            return func(tensor, *args, **kwargs)
    return wrapper

@mps_compatible
def broadcast(tensor, src=0):
    return dist.broadcast(tensor, src=src)

@mps_compatible
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    return dist.all_reduce(tensor, op=op)

@mps_compatible
def all_gather(tensor):
    return dist.all_gather(tensor)

# @mps_compatible
# def reduce_scatter(tensor):
#     return dist.reduce_scatter(tensor)

# @mps_compatible
# def reduce(tensor):
#     return dist.reduce(tensor)

# @mps_compatible
# def gather(tensor):
#     return dist.gather(tensor)