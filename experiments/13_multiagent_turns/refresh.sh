#!/bin/bash
# Re-extract the calls of the finished episodes and of the newest copy of the live buffer, and relabel them.
cd "$(dirname "$0")"
PY=/resnick/groups/Safety_First/aquisper/mamubench/.venv/bin/python
P=/resnick/scratch/aquisper/mamu_pilot
$PY extract_calls.py $P/runs_pilot $P/calls/pilot 2>&1 | tail -6
DB=$(ls -t $P/buffer_snapshots/*.db 2>/dev/null | head -1)
[ -n "$DB" ] && $PY extract_calls.py $DB $P/calls/team_live --name ${1:-team_e1_attempt2} 2>&1 | tail -3
python3 labels.py $P/calls/pilot $P/calls/team_live > $P/calls.csv && wc -l $P/calls.csv
