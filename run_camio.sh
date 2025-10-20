#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables for proper audio operation
export PULSE_RUNTIME_PATH=/run/user/$(id -u)/pulse
export ALSA_PCM_CARD=0
export ALSA_PCM_DEVICE=0

# Run the application with xvfb-run for automatic virtual display
xvfb-run -a -s "-screen 0 1024x768x24" python camio_opt.py --input '/home/user/camio/models/CnapMap/CnapFirstFloor.json'