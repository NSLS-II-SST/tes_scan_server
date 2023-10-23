#!/usr/bin/env bash

if pgrep cringe &> /dev/null; then
    echo "Found Cringe"
else
    gnome-terminal --tab -t cringe -- cringe -F /home/xf07id1/cringe_config/20220323_8col_30row_nsls_50mK.pkl
fi

if pgrep dastard &> /dev/null; then
    echo "Found Dastard"
else
    gnome-terminal --tab -t dastard -- dastard
fi

if pgrep dcom &> /dev/null; then
    echo "Found Dcom"
else
    gnome-terminal --tab -t dcom -- dcom
fi

if pgrep adr_gui &> /dev/null; then
    echo "Found ADR_Gui"
else
    gnome-terminal --tab -t adrgui -- adr_gui2
fi

