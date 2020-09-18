A server to stand between a beamline (eg the spec computer at SSRL 10-1) and DASTARD/mass.

The beamline talks to ScanServer, ScanServer talks to Dastard and mass. Dastard writes OFF files. The TES must be already tuned, dastard must already have projectors loaded.

## Important directory/file locations

  * `base_log_dir/beamtime_id/scan00000/log.json`
    * human/computer readable log
    * also copied into the .off directory
  * `base_log_dir/beamtime_id/scan00000/plots`
    * realtime plots
  * `base_log_dir/beamtime_id/scan00000/data`
    * computer readable data
  * cals in the .off directory

