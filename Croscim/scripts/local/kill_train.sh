#!/bin/bash
# Kill training process
cd /home/malegu/4D-MLG/Croscim
pkill -9 -f "python main.py xp=SST/multires"
rm -f process.pid
echo "OK : All processes killed"
