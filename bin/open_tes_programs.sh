#!/usr/bin/env bash

# List of approved programs
approved_programs=("cringe" "dastard" "dcom" "adr_gui")

# Function to check if a program is running and start it if not
start_program_if_not_running() {
    local program=$1
    local command=$2

    if pgrep "$program" &> /dev/null; then
        echo "Found $program"
    else
        gnome-terminal --tab -t "$program" -- $command
    fi
}

# List of programs to open
programs_to_open=("$@")

# Iterate over the list of programs to open
for program in "${programs_to_open[@]}"; do
    case $program in
        cringe)
            start_program_if_not_running "cringe" "cringe -F /home/xf07id1/cringe_config/20220323_8col_30row_nsls_50mK.pkl"
            ;;
        dastard)
            start_program_if_not_running "dastard" "dastard"
            ;;
        dcom)
            start_program_if_not_running "dcom" "dcom"
            ;;
        adr_gui)
            start_program_if_not_running "adr_gui" "adr_gui2"
            ;;
        *)
            echo "Program $program is not approved to be started."
            ;;
    esac
done
