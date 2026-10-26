#!/usr/bin/env bash
# 10B export: browser app vs desktop app (offscreen) vs notebook 4, same env, same saved grains.
set -e; cd "$(dirname "$0")/.."; export MPLBACKEND=Agg
python tools/web_flow.py /tmp/webout
QT_OUT=/tmp/qtout python -m pytest -q tools/qt_flow.py -p no:cacheprovider
rm -rf /tmp/nbout
python - <<'P'
import sys; sys.path.insert(0, "tools"); from web_flow import seed; seed("/tmp/nbout")
P
python tools/nb4_run.py /tmp/nbout >/dev/null
s() { [ -f "$1" ] && md5sum < "$1" | cut -c1-10 || echo "----------"; }
for f in $(ls /tmp/webout/amics/10B /tmp/qtout/amics/10B /tmp/nbout/amics/10B | grep 10B_ | sort -u); do
  a=$(s /tmp/webout/amics/10B/$f); b=$(s /tmp/qtout/amics/10B/$f); c=$(s /tmp/nbout/amics/10B/$f)
  printf "%-28s %s %s %s %s\n" $f $a $b $c $([ $a = $b ] && [ $a = $c ] && echo SAME || echo DIFF); done
