#!/bin/bash
# Initialize ZenML
zenml init
# Optional: Start ZenML server or any other service
zenml up # --docker
# Keep container running (if the above commands do not do this)
tail -f /dev/null
