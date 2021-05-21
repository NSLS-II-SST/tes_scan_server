from scan_server.rpc_server import get_dispatch_from
from scan_server.dastard_client import DastardClient
from os.path import join

def fakeMethodMaker(f):
    def _inner(self, *args, **kwargs):
        if self.verbose: print(f"Method {f} called with {args} and {kwargs}")
        return True
    return _inner

class FakeDastardClient():
    def __init__(self, verbose=False):
        self.off_filename = None
        self.verbose = verbose
        dispatch = get_dispatch_from(DastardClient)
        functions = dispatch.keys()
        for f in functions:
            if not hasattr(self, f):
                setattr(self.__class__, f, fakeMethodMaker(f))

    def start_file(self, ljh22, off, path=None):
        if self.verbose: print(f"Method start_file called with {ljh22}, {off}, and path: {path}")
        if path is not None:
            self.off_filename = join(path, "tmp_chan1.off")
        else:
            self.off_filename = join("/tmp", "tmp_chan1.off")
        return self.off_filename

    def get_data_path(self):
        if self.verbose: print("Method get_data_path called")
        return self.off_filename
