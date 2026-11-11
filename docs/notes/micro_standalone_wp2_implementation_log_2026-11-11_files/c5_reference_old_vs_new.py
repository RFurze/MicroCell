"""Old c5 reference against the re-made one: stdlib only (npz = zip of .npy)."""
import io, json, struct, sys, zipfile, ast

def load(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            b = z.read(n)
            assert b[:6] == b"\x93NUMPY"
            major = b[6]
            if major == 1:
                hl = struct.unpack("<H", b[8:10])[0]; off = 10
            else:
                hl = struct.unpack("<I", b[8:12])[0]; off = 12
            hdr = ast.literal_eval(b[off:off + hl].decode("latin1"))
            data = b[off + hl:]
            out[n[:-4]] = (hdr, data)
    return out

def js(rec):
    hdr, data = rec
    d = hdr["descr"]
    assert d.startswith("<U"), d
    return json.loads(data.decode("utf-32-le"))

def arr(rec):
    hdr, data = rec
    assert hdr["descr"] == "<f8", hdr
    return struct.unpack("<%dd" % (len(data) // 8), data)

old, new = load(sys.argv[1]), load(sys.argv[2])
rows = ["member\tname\told\tnew\trel"]
assert sorted(old) == sorted(new), (sorted(old), sorted(new))
for m in sorted(old):
    if m.endswith("_json"):
        a, b = js(old[m]), js(new[m])
        if not isinstance(a, dict):
            if a != b: rows.append(f"{m}\t-\t{a!r}\t{b!r}\t")
            continue
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            if x != y and not (isinstance(x, float) and isinstance(y, float) and x != x and y != y):
                rel = ""
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                    den = max(abs(x), abs(y))
                    rel = repr(abs(x - y) / den) if den else ""
                rows.append(f"{m}\t{k}\t{x!r}\t{y!r}\t{rel}")
    else:
        a, b = arr(old[m]), arr(new[m])
        if a != b:
            den = max(max(abs(v) for v in a), max(abs(v) for v in b))
            dif = max(abs(p - q) for p, q in zip(a, b) if p == p and q == q)
            rows.append(f"{m}\t(array, {len(a)} values)\t\t\t{dif / den!r}")
rows.append(f"# {len(old)} members; {len(rows) - 1} differing entries")
open(sys.argv[3], "w", newline="\n").write("\n".join(rows) + "\n")
print("\n".join(r[:230] for r in rows))
