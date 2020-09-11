# here we implment something like json rpc
# but to keep the requirements on the "client" lower
# our responses are simpler than json text, and we don't require the "jsonrpc"="2.0" key

import time
import os
import socket
import json



DISPATCH = {}

DISPATCH["add"] = lambda a,b: a+b
DISPATCH["echo"] = lambda s: s

def call_method_from_data(data, dispatch):
    try:
        d = json.loads(data)
    except Exception as e:
        return None, None, None, None, f"JSON Parse Exception: {e}"
    _id = d.get("id", -1)
    if "method" not in d.keys():
        return _id, None, None, None, f"method key does not exist"
    method_name = d["method"]
    if "params" not in d.keys():
        return _id, method_name, None, None, f"params key does not exist"   
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
        return _id, method_name, args, None, f"Calling Exception: method={method_name}: {e}"

def make_simple_response(_id, result, error):
    if result is None:
        response = f"Error: {error}"
    else:
        resposne = f"{result}"
    return response

def handle_one_message(socket, dispatch, verbose):
    data = socket.recv(2**12)
    if verbose:
        print(f"got: {data}")
    _id, method_name, args, result, error = call_method_from_data(data, dispatch)
    if verbose:
        print(f"id: {_id}, method_name: {method_name}, args: {args}, result: {result}, error: {error}")
    response = make_simple_response(_id, method_name, args, result, error)
    socket.sendall(response.encode())




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

def start(address="localhost", port=4000):
    print("TES Scan Server")
    print(f"{address}:{port}")
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind((address, port))
    # become a server socket
    serversocket.listen(1)

    try:
        while True:
            # accept connections from outside
            (clientsocket, address) = serversocket.accept()
            print(f"connection from {address}")
            while True:
                handle_one_message(clientsocket, DISPATCH, verbose=True)   
    except KeyboardInterrupt:
        print("\nCtrl-C detected, shutting down")
        return
    
     