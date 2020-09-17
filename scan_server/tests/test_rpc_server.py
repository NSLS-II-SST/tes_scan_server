from scan_server import rpc_server
import json

def test_call_method_from_data():
    dispatch = {}
    dispatch["add"] = lambda a,b: a+b
    dispatch["echo"] = lambda s: s

    data = json.dumps({"method": "add", "params": [1, 2]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert _id == -1
    assert method_name == "add"
    assert result == 3
    assert args == [1, 2]
    assert error is None

    data = json.dumps({"method": "echo", "params": [6]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert _id == -1
    assert method_name == "echo"
    assert result == 6
    assert args == [6]
    assert error is None

    data = json.dumps({"method": "not_exist", "params": [6]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert error == "Method 'not_exist' does not exit, valid methods are ['add', 'echo']"
   
    data = json.dumps({"method": "echo", "params": 6})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert error == "args must be a list, instead it is 6"

    data = json.dumps({"method": "echo"})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert error == "params key does not exist"

    data = json.dumps({"params": [1, 2]})
    _id, method_name, args, result, error = rpc_server.call_method_from_data(data, dispatch)
    assert error == "method key does not exist"