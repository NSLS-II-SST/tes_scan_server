Installation: `pip install -e git+ssh://git@bitbucket.org/nist_microcal/scan_server.git#egg=scan_server`

A server to stand between a beamline (eg the spec computer at SSRL 10-1) and DASTARD/mass.

The beamline talks to ScanServer, ScanServer talks to Dastard and mass. Dastard writes OFF files. The TES must be already tuned, dastard must already have projectors loaded.

## Important directory/file locations

  * `base_user_output_dir/beamtime_id/logs/scan00000.json`
  * `base_user_output_dir/beamtime_id/logs/calibration0000.json`
  * `base_user_output_dir/beamtime_id/scan0000/plots` 
  * `base_user_output_dir/beamtime_id/scan0000/data` 
  * `off_dir/logs/scan0000/scan00000.json`
  * `off_dir/logs/scan0000/calibration0000.json`

 
