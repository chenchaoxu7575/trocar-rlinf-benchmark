import sqlite3
c = sqlite3.connect("/out/gfx.sqlite")
def strid(i):
    r = c.execute("select value from StringIds where id=?", (i,)).fetchone()
    return r[0] if r else None
cols = [d[1] for d in c.execute("PRAGMA table_info(NVTX_EVENTS)")]
tcol = "text" if "text" in cols else None
# collect ranges of interest
want = ["run_training", "interact", "generate_one_epoch", "predict", "env_interact_step", "isaaclab.sim_step"]
ev = {w: [] for w in want}
for row in c.execute("select start, end, textId, text from NVTX_EVENTS where end is not null"):
    st, en, tid, txt = row
    name = txt if txt else strid(tid)
    if not name:
        continue
    name = name.lstrip(":")
    if name in ev:
        ev[name].append((st, en))
for w in ev:
    ev[w].sort()
def dur(lst):
    return sum(e - s for s, e in lst) / 1e9
print("instances:", {w: len(ev[w]) for w in want})

# rollout phase per step = :interact (env worker wraps the whole rollout ping-pong)
roll = ev["interact"]        # expect 2
train = ev["run_training"]   # expect 2
if len(roll) >= 2 and len(train) >= 2:
    print("\n=== per-RL-step timeline (ns->s) ===")
    steps = []
    for i in range(min(len(roll), len(train))):
        rs, re = roll[i]
        ts, te = train[i]
        steps.append((rs, re, ts, te))
    # full step period: from rollout start[i] to rollout start[i+1]
    for i, (rs, re, ts, te) in enumerate(steps):
        roll_w = (re - rs) / 1e9
        train_w = (te - ts) / 1e9
        gap_r2t = (ts - re) / 1e9              # rollout end -> train start
        if i + 1 < len(steps):
            period = (steps[i + 1][0] - rs) / 1e9
            gap_t2r = (steps[i + 1][0] - te) / 1e9   # train end -> next rollout start
        else:
            period = (te - rs) / 1e9
            gap_t2r = None
        print("step%d: rollout=%.1fs  gap_r2t=%.1fs  train=%.1fs  gap_t2r=%s  period=%.1fs"
              % (i, roll_w, gap_r2t, train_w, ("%.1fs" % gap_t2r) if gap_t2r is not None else "n/a", period))
    # average over the first (complete) step for proportions
    rs, re, ts, te = steps[0]
    period = (steps[1][0] - rs) / 1e9
    roll_w = (re - rs) / 1e9
    train_w = (te - ts) / 1e9
    gap_r2t = (ts - re) / 1e9
    gap_t2r = (steps[1][0] - te) / 1e9
    print("\n=== ONE RL step (step0 -> step1 boundary) ===")
    for lbl, v in [("rollout", roll_w), ("gap rollout->train", gap_r2t), ("train", train_w), ("gap train->rollout", gap_t2r)]:
        print("   %-20s %7.1f s   %5.1f%%" % (lbl, v, 100 * v / period))
    print("   %-20s %7.1f s   100.0%%" % ("TOTAL RL step", period))
c.close()
