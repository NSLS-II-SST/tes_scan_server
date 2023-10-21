from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
import io
import os


@dataclass_json
@dataclass
class BaseScan():
    var_name: str
    var_unit: str
    scan_num: int
    beamtime_id: str
    sample_id: int
    sample_desc: str
    extra: dict
    data_path: str
    point_extras: dict = field(default_factory=dict)
    var_values: List[float] = field(default_factory=list)
    epoch_time_start_s: List[int] = field(default_factory=list)
    epoch_time_end_s: List[int] = field(default_factory=list)
    _ended: bool = field(default=False)

    def point_start(self, scan_var, epoch_time_s, extra=None):
        assert not self._ended
        assert len(self.epoch_time_start_s) == len(self.epoch_time_end_s)

        self.var_values.append(float(scan_var))
        if extra is not None and extra != {}:
            assert isinstance(extra, dict)
            idx = str(len(self.epoch_time_start_s))
            self.point_extras[idx] = extra
        # print(f"{self.epoch_time_start_s}")
        self.epoch_time_start_s.append(epoch_time_s)
        # print(f"{self.epoch_time_start_s}")

    def point_end(self, epoch_time_s):
        assert len(self.epoch_time_start_s) - 1 == len(self.epoch_time_end_s)
        self.epoch_time_end_s.append(epoch_time_s)

    def write_experiment_state_file(self, f, header):
        if header:
            f.write("# unixnano, state label\n")
        for i, (start, end) in enumerate(zip(self.epoch_time_start_s, self.epoch_time_end_s)):
            label = f"SCAN{self.scan_num}_{i}"
            f.write(f"{int(start*1e9)}, {label}\n")
            f.write(f"{int(end*1e9)}, PAUSE\n")

    def experiment_state_file_as_str(self, header):
        with io.StringIO() as f:
            self.write_experiment_state_file(f, header)
            return f.getvalue()

    def end(self):
        assert not self._ended
        self._ended = True

    def description_str(self):
        return f"scan{self.scan_num} sample{self.sample_id} beamtime_id{self.beamtime_id}"

    def __repr__(self):
        return f"<Scan num{self.scan_num} beamtime_id{self.beamtime_id} npts{len(self.var_values)}>"

    def to_disk(self, filename, overwrite=False):
        if not overwrite:
            assert not os.path.isfile(filename)
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            return cls.from_json(f.read())


@dataclass_json
@dataclass
class DataScan(BaseScan):
    cal_number: int = -1
    calibration: bool = field(default=False)


@dataclass_json
@dataclass
class CalibrationScan(BaseScan):
    routine: str = "none"
    calibration: bool = field(default=True)
