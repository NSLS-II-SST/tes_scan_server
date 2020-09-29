from scan_server import rpc_server
import json
import threading
import tempfile
import time
import socket

def test_call_method_from_data():
    dispatch = {}
    dispatch["add"] = lambda a,b: a+b
    dispatch["echo"] = lambda s: s
    no_traceback_error_types = []
    data = json.dumps({"method": "add", "params": [1, 2]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert _id == -1
    assert method_name == "add"
    assert result == 3
    assert args == [1, 2]
    assert error is None

    data = json.dumps({"method": "echo", "params": [6]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert _id == -1
    assert method_name == "echo"
    assert result == 6
    assert args == [6]
    assert error is None

    data = json.dumps({"method": "not_exist", "params": [6]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert error == "Method 'not_exist' does not exit, valid methods are ['add', 'echo']"
   
    data = json.dumps({"method": "echo", "params": 6})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert error == "args must be a list, instead it is 6"

    data = json.dumps({"method": "echo"})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert error == "params key does not exist"

    data = json.dumps({"params": [1, 2]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch, no_traceback_error_types)
    assert error == "method key does not exist"

def test_with_real_socket():
    # following https://www.devdungeon.com/content/unit-testing-tcp-server-client-python
    dispatch = {}
    dispatch["no_args"] = lambda: None
    dispatch["add"] = lambda a,b: a+b
    dispatch["echo"] = lambda s: s
    class TestError(Exception):
        pass
    def throw_test_error():
        raise TestError("test error")
    def throw_keyboard_interrupt():
        raise KeyboardInterrupt()
    dispatch["throw_test_error"] = throw_test_error
    dispatch["throw_keyboard_interrupt"] = throw_keyboard_interrupt

    tempfname = tempfile.mktemp()
    tempf = open(tempfname, "w")
    def start():
        rpc_server.start("localhost", 4000, dispatch, verbose=False, log_file=tempf,
        no_traceback_error_types = [])

    server_thread = threading.Thread(target=start)
    server_thread.start()

    time.sleep(.01) # make sure the thread has time to start up

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.settimeout(.1)
    client.connect(('localhost', 4000))
    client.send(b"""{"method": "echo", "params": ["yo"]}""")
    assert client.recv(2**14) == b"yo"
    client.send(b"""{"method": "echo", "params": ["yo"], "extra": 4}""")
    assert client.recv(2**14) == b"yo"
    client.send(b"""{"method": "add", "params": [4.2, 5.5]}""")
    assert client.recv(2**14) == b"9.7"
    client.send(b"""{"method": "no_args", "params": []}""")
    assert client.recv(2**14) == b"None"
    client.send(b"""{"method": "echo", "params": ["yo", "yo"]}""")
    assert client.recv(2**14) == b"Error: Calling Exception: method=echo: <lambda>() takes 1 positional argument but 2 were given"
    client.send(b"""{"method": "faust", "params": []}""")
    assert client.recv(2**14) == b"Error: Method 'faust' does not exit, valid methods are ['no_args', 'add', 'echo', 'throw_test_error', 'throw_keyboard_interrupt']"
    client.send(b"""{"method": "throw_test_error", "params": []}""")
    assert client.recv(2**14) == b'Error: Calling Exception: method=throw_test_error: test error'
    client.send(b"""YO ADRIAN""")
    assert client.recv(2**14) == b'Error: JSON Parse Exception: Expecting value: line 1 column 1 (char 0)'    
    client.send(b"""{YO, [] ADRIAN}""")
    assert client.recv(2**14) == b'Error: JSON Parse Exception: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)'

    assert server_thread.is_alive()

    # the server should run until ctrl-c, so we emulate ctrl-c with throw_test_error
    # then wait a bit, then check that the thread is not alive
    client.send(b"""{"method": "throw_keyboard_interrupt", "params": []}""")
    time.sleep(0.01) 
    assert not server_thread.is_alive()