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

 ## Intentions

 # Data Outputs
 1. In real time, PFY_realtime comes from `TESScanner.roi_get_counts` via `TESScanner._data`
 1.b. We could add a RIXS_realtime easily if a beamline system wanted it.
 2. Immediatley after a scan, PFY_prelim and RIXS_prelim come from `TESScanner.quick_post_process` which relites on `TESScanner._data`. For example even if the dirft correction plan want a later calibration, just use the current calibration.
 3. Sometime after a scan, PFY_final and RIXS final come from `post_process`, and may depend on calibrations that occur after the scan. This reprocessed from the off files and scan logs.

 Perhaps beamline visualization will probably use 1 and/or 2. 3 is what the users take home, but 

