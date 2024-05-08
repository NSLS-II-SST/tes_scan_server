#!/usr/bin/env bash

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

