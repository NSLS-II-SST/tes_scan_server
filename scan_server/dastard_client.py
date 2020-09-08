import requests
import json
import itertools


class DastardClient():
    def __init__(self, url):
        self.url = url
        self._id_iter = itertools.count()


    def _message(self, method, params):
        return dict(id=next(self._id_iter),
                    params=[params],
                    method=method)

    def _call(self, method, params):
        msg = self._message(method, params)
        reponse = requests.post(self.url, json=msg).json()
        assert response["id"] == msg["id"]
        assert response["error"] is None
        print(f"{response=}")
        return response

    def start_file(self):
        params = {"Request": "Start",
        "WriteLJH22": False,
        "WriteLJH3": False,
        "WriteOFF": True,
        }
        response = self._call("SourceControl.WriteControl", params)

    def stop_file(self):
        payload = {"Request": "Stop"}
        response = self._call("SourceControl.WriteControl", params)

    def send_experiment_state(self, state):
        pass