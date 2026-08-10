#!/bin/bash
# Reboot our test server on sparky with a wiped disk cache.
# $1 = extra server args (quoted string, may be empty), $2 = env prefix string
set -u
ssh claude@sparky.defense.lan "pkill -x pulsar-server; sleep 3; rm -rf /home/claude/.cache/ds4/kv-v5mx4-0731-mmqaligned; cd ~/idx-engine && (env $2 PULSAR_LOCK_FILE=/home/claude/nsystmp/ds4srv.lock setsid nohup ./pulsar-server -m /srv/models/v5mx4-0731-mmqaligned.gguf --cuda --port 8099 $1 > /home/claude/nsystmp/srv_cycle.log 2>&1 < /dev/null &); sleep 30; echo up"
curl -s -m 5 http://sparky.defense.lan:8099/version > /dev/null && echo reachable
