import re, pathlib, sys

path = pathlib.Path("app.py")
s = path.read_text()

replacement = '''# --- Conditional availability check ---
COND_META_PATH = "data/cond_cdf_meta.json"
COND_DIR = "data"

cond_available = False
cond_reason = ""

if os.path.exists(COND_META_PATH):
    try:
        cm = json.load(open(COND_META_PATH))
        bucket_files = cm.get("bucket_files", [])
        if not bucket_files or len(bucket_files) != 10:
            cond_reason = "cond meta exists but bucket_files list is missing/invalid"
        else:
            missing = []
            for fn in bucket_files:
                p = os.path.join(COND_DIR, fn)
                if not os.path.exists(p):
                    missing.append(fn)
            if missing:
                cond_reason = f"missing bucket files (likely Git LFS not pulled): {missing[:3]}{'...' if len(missing)>3 else ''}"
            else:
                cond_available = True
    except Exception as e:
        cond_reason = f"failed to read cond meta: {e}"
else:
    cond_reason = "missing data/cond_cdf_meta.json"

use_cond = st.checkbox("Use regime-conditional CDF", value=False, key="use_cond")

if use_cond and not cond_available:
    st.error("Regime-conditional CDF requested but files are not available.")
    st.write(cond_reason)
    st.stop()

if not cond_available:
    st.caption("Regime-conditional CDF not available on this deployment.")
    st.caption(cond_reason)
'''

pattern = re.compile(
    r'cond_available\s*=\s*.*?\n'
    r'(?:.*\n){0,80}?'
    r'use_cond\s*=\s*st\.checkbox\([^\n]*\)\s*\n',
    re.DOTALL
)

m = pattern.search(s)
if not m:
    print("PATCH FAILED: Could not find the cond_available/use_cond block.")
    sys.exit(1)

path.write_text(s[:m.start()] + replacement + s[m.end():])
print("Patched app.py successfully.")
