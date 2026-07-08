import sqlite3
c = sqlite3.connect("/out/s36.sqlite")
def strid(i):
    r = c.execute("select value from StringIds where id=?", (i,)).fetchone()
    return r[0] if r else None
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print("ALL tables:", [t for t in tabs if any(k in t.upper() for k in ["OSRT","RUNTIME","SCHED","API","GENERIC"])])
# pid map by high bits of globalPid
gp2pid = {}
for gp, pid, name in c.execute("select globalPid, pid, name from PROCESSES"):
    gp2pid[gp >> 24] = (pid, name)
def role(tid):
    return gp2pid.get(tid >> 24, (tid >> 24, "?"))
for t in ["CUPTI_ACTIVITY_KIND_RUNTIME", "OSRT_API"]:
    if t not in tabs:
        print("(no table %s)" % t); continue
    cols = [d[1] for d in c.execute("PRAGMA table_info(%s)" % t)]
    print("\n===== %s  cols=%s =====" % (t, cols))
    rows = list(c.execute("select globalTid, nameId, (end-start) as dur from %s" % t))
    agg = {}
    for tid, nid, dur in rows:
        pid, pn = role(tid)
        key = (pid, pn, strid(nid))
        a = agg.setdefault(key, [0, 0])
        a[0] += dur; a[1] += 1
    for (pid, pn, nm), (tot, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:16]:
        print("   pid=%-6s %-16s %-26s %9.1f ms  x%d" % (pid, pn, nm, tot / 1e6, n))
c.close()
