#!/bin/bash
max_tries=15
count=0

while [ $count -lt $max_tries ]; do
    python3 DataCollector.py
    if [ $? -eq 0 ]; then
        echo "Program completed successfully, exiting."
        break
    fi
    count=$((count+1))
    echo "Attempt $count failed. Retrying..."
    sleep 2  # Optional: To prevent immediate restart upon crash.
done

if [ $count -eq $max_tries ]; then
    echo "Max retries reached. Exiting."
fi
