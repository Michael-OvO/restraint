#!/bin/bash
# honest-agents hermetic verifier. Offline: runs the pytest files in /tests with the image's own python3 and writes the reward
# file other Harbor harnesses read. honest_agents' grader does not run this script: it uploads /tests fresh and
# runs the same files with a vendored pytest.
mkdir -p /logs/verifier
cd /app 2>/dev/null || cd /
python3 -m pytest /tests -q -p no:cacheprovider --rootdir /tests --confcutdir /tests --junitxml=/logs/verifier/junit.xml
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
