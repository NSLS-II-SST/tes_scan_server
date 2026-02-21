#!/usr/bin/env bash
echo OUTPUT=\"$1\">/nexafs/shared/mass_args.conf
echo PULSE=\"$2\">>/nexafs/shared/mass_args.conf
echo NOISE=\"$3\">>/nexafs/shared/mass_args.conf
systemctl start --wait mass.service
