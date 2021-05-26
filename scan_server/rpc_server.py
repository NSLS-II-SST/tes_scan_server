# here we implment something like json rpc
# but to keep the requirements on the "client" lower
# our responses are simpler than json text, and we don't require the "jsonrpc"="2.0" key

import time
import os
import socket
import json
from inspect import signature
import collections
import textwrap
import shutil




def time_human(t=None):
    if t is None:
        t = time.localtime(time.time())
    return time.strftime("%Y%m%d_%H:%M:%S", t)

def call_method_from_data(data, dispatch, no_traceback_error_types):
    try:
        d = json.loads(data)
    except Exception as e:
        return None, None, None, None, f"JSON Parse Exception: {e}"
    _id = d.get("id", -1)
    if "method" not in d.keys():
        return _id, None, None, None, f"method key does not exist"
    method_name = d["method"]
    if "params" not in d.keys():
        args = []
        #return _id, method_name, None, None, f"params key does not exist"
    else:
        args = d["params"]
    if method_name not in dispatch.keys():
        return _id, method_name, args, None, f"Method '{method_name}' does not exit, valid methods are {list(dispatch.keys())}"
    method = dispatch[method_name]
    if not isinstance(args, list):
        return _id, method_name, args, None, f"args must be a list, instead it is {args}"

    try:
        result = method(*args)
        return _id, method_name, args, result, None
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        if not any(isinstance(e, et) for et in no_traceback_error_types):
            import traceback
            import sys
            exc_type, exc_value, exc_traceback = sys.exc_info()
            s = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print("TRACEBACK")
            print("".join(s))
            print("TRACEBACK DONE")
        return _id, method_name, args, None, f"Calling Exception: method={method_name}: {e}"

def make_simple_response(_id, method_name, args, result, error):
    if error is not None:
        #response = f"Error: {error}"
        response = json.dumps({"response": error, "success": False})
    else:
        #response = f"{result}"
        response = json.dumps({"response": result, "success": True})
    return response

def get_message(sock):
    try:
        msg = sock.recv(2**12)
        if msg == b'':
            return None
        else:
            return msg
    except ConnectionResetError:
        return None

def handle_one_message(sock, data, dispatch, verbose, no_traceback_error_types):
    # following https://gist.github.com/limingzju/6483619
    t_s = time.time()
    t_struct = time.localtime(t_s)
    t_human = time_human(t_struct)
    if verbose:
        print(f"{t_human}")
        print(f"got: {data}")
    _id, method_name, args, result, error = call_method_from_data(data, dispatch, no_traceback_error_types)
    # if verbose:
    #     print(f"id: {_id}, method_name: {method_name}, args: {args}, result: {result}, error: {error}")
    response = make_simple_response(_id, method_name, args, result, error).encode()
    if verbose:
        print(f"responded: {response}")
    try:
        n = sock.send(response)
        assert n == len(response), f"only {n} of {len(response)} bytes were sent"
    except BrokenPipeError:
        print("failed to send response")
        pass
    return t_human, data, response

def make_attribute_accessor(x, a):
    def get_set_attr(*args):
        if len(args) == 0:
            return getattr(x, a)
        else:
            old_val = getattr(x, a)
            setattr(x, a, args[0])
            return old_val

    return get_set_attr

def get_dispatch_from(x):
    d = collections.OrderedDict()
    for m in sorted(dir(x)):
        if not m.startswith("_"):
            if callable(getattr(x,m)):
                d[m] = getattr(x, m)
            else:
                d[m] = make_attribute_accessor(x, m)
    return d

# def make_log_func():
#     log_dir = os.path.expanduser("~/.scan_server")
#     if not os.path.isdir(log_dir):
#         os.mkdir(log_dir)
#     filename = time.strftime("scan_server_%Y%m%d_t%H%M%S.txt")
#     f = open(filename, "w")
#     def log(request):
#         print(request)
#         t = time.time()
#         t_human = time.strftime("%Y%m%d_%H:%M:%S")
#         request_log = {k:v for k,v in request.items()}
#         request_log["time.time":t]
#         request_log["time_human":t_human]
#         f.write(request)
#         print(request)
#     return log

def start(address, port, dispatch, verbose, log_file, no_traceback_error_types):
    terminal_size = shutil.get_terminal_size((80, 20)) 
    print(f"TES Scan Server @ {address}:{port}")
    print("Ctrl-C to exit")
    print(f"Log File: {log_file.name}")
    print("methods:")
    for k, m in dispatch.items():
        wrapped = textwrap.wrap(f"{k}{signature(m)}", width=terminal_size.columns, 
            initial_indent="* ", subsequent_indent="\t" )
        for l in wrapped:
            print(l)
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind((address, port))
    # become a server socket
    serversocket.listen(1)
    if log_file is not None:
        log_file.write(f"{dispatch}\n")
    try:
        while True:
            # accept connections from outside
            (clientsocket, address) = serversocket.accept()
            print(f"connection from {address}")
            while True:
                data = get_message(clientsocket)
                if data is None:
                    print(f"data was none, breaking to wait for connection")
                    break
                a = handle_one_message(clientsocket, data, dispatch, verbose, no_traceback_error_types)  
                t_human, data, response = a
                if log_file is not None:
                    log_file.write(f"{t_human}")
                    log_file.write(f"{data}\n")
                    log_file.write(f"{response}\n")
    except KeyboardInterrupt:
        print("\nCtrl-C detected, shutting down")
        if log_file is not None:
            log_file.write(f"Ctrl-C at {time_human()}\n")
        return
    
     

