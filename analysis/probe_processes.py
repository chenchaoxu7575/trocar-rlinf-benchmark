import sqlite3
c = sqlite3.connect("/out/s36.sqlite")
def strid(i):
    r = c.execute("select value from StringIds where id=?", (i,)).fetchone()
    return r[0] if r else None
for gp, pid, name in c.execute("select globalPid, pid, name from PROCESSES"):
    try:
        kn = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=?", (gp,)).fetchone()[0]
    except Exception:
        kn = 0
    if kn == 0:
        continue
    topk = []
    for sid, cnt in c.execute("select shortName, count(*) cnt from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? group by shortName order by cnt desc limit 5", (gp,)):
        topk.append((("" if sid is None else (strid(sid) or ""))[:44], cnt))
    print("pid=%s  name=%s  kernels=%s" % (pid, name, kn))
    for k, n in topk:
        print("        %6d  %s" % (n, k))
c.close()
