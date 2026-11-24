#!/usr/bin/env bash
# bash autoresearch/launch_mac_varloop.sh RUN_DIR [HOURS=8]: a track-C autoresearch run for the variable-loop Inkling 2 Mini
# at 5 passes, starting from engines/inkling-2-mini_m5max/offline-c (varloop, 2DSD drafter). As launch_mac.sh, but with
# this model's checkpoint, reference, baselines and prompt (autoresearch/prompts/first_C_mac_varloop.md).
set -uo pipefail
zoo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="${1:?run dir}"; hours="${2:-8}"
model="${MODEL:-claude-fable-5-2}"; resumes="${MAX_RESUMES:-20}"
export MK_MODEL=thinkingmachines/TML-d24-varloop-5pass
export MK_MODEL_DIR="${MK_MODEL_DIR:-$HOME/Models/Inkling/TML-d24-varloop-5pass}"
export MK_REF_DIR="${MK_REF_DIR:-$HOME/mk-ref/tml-d24-varloop-5pass}"
log() { echo "$(date '+%F %T') $*" | tee -a "$run.launch.log"; }
for f in ids.npy meta.json logits64.npy long_positions.npy long_logits64.npy; do
  [ -s "$MK_REF_DIR/$f" ] || { log "no $MK_REF_DIR/$f"; exit 2; }
done
bash "$zoo/scripts/eval_setup_mac.sh" 2>&1 | tee -a "$run.launch.log" || { log "eval_setup_mac.sh failed"; exit 3; }
if [ ! -d "$run" ]; then
  "$zoo/.venv-mac/bin/python" "$zoo/autoresearch/start.py" "$run" --model "$MK_MODEL" --card m5max --track C --hours "$hours" \
    --model-dir "$MK_MODEL_DIR" --ref-dir "$MK_REF_DIR" --engine "$zoo/engines/inkling-2-mini_m5max/offline-c" 2>&1 | tee -a "$run.launch.log"
  # the varloop prompt in place of the Mac one, with start.py's placeholders filled the same way
  "$zoo/.venv-mac/bin/python" - "$run" "$zoo" <<'PY'
import json, sys
from pathlib import Path
run, zoo = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((run / "run.json").read_text())
refs = cfg["reference_lines"]
baseline = "; ".join(f"{r['name']}: {r['value']} us per token" + (f" ({r['note']})" if r["note"] else "") for r in refs[:-1])
t = (zoo / "autoresearch/prompts/first_C_mac_varloop.md").read_text()
t = t.replace("<model>", cfg["model"]).replace("<card>", cfg["card"]).replace("<hours>", f"{cfg['hours']:g}")
t = t.replace("<baseline>", baseline).replace("<mk>", str(run / "bin" / "mk"))
(run / "PROMPT.md").write_text(t)
PY
fi
mv "$run.launch.log" "$run/launch.log" 2>/dev/null; exec >> "$run/launch.log" 2>&1
log "run $run, $hours h, agent $model"
time_left() { PATH="$run/bin:$PATH" mk board 2>/dev/null | sed -n 's/.* \([0-9.]*\) left;.*/\1/p' | head -1; }
cd "$run/workspace" || exit 4
caffeinate -dims -w $$ &
for i in $(seq 0 "$resumes"); do
  if [ "$i" = 0 ]; then
    log "agent session 0"
    claude -p "$(cat ../PROMPT.md)" --model "$model" --dangerously-skip-permissions --output-format stream-json --verbose \
      > "$run/agent-0.jsonl" 2>&1
  else
    left="$(time_left)"; log "agent stopped; board says ${left:-?} h left"
    awk -v l="${left:-0}" 'BEGIN { exit !(l > 0.1) }' || break
    claude -p --continue "You stopped, but the run still has ${left} hours on the board. Continue the run as PROMPT.md says: read your notes and \`mk board\`, decide the next experiment, and keep going until the time is used up." \
      --model "$model" --dangerously-skip-permissions --output-format stream-json --verbose > "$run/agent-$i.jsonl" 2>&1
  fi
done
log "run finished: $(PATH="$run/bin:$PATH" mk board 2>/dev/null | head -1)"
