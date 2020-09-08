from scan_server import TESScanner, Scan, DastardClient

class MockClient(DastardClient):
    def _call(self, method, request):
        return None

def test_tes_scanner():
    scanner = TESScanner(client = MockClient(""), pulse_trigger_msg = None, noise_trigger_msg = None,
    beamtime_id ="test")
    scanner.file_start()

def test_scan():
    scan = Scan(["mono_ev","temp_K"], 1, "fake_beamtime", 12)