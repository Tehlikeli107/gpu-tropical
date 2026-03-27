import subprocess, os
cwd = r"C:\Users\salih\Desktop\gpu-tropical"
env = os.environ.copy()
env.update({"GIT_AUTHOR_NAME": "salih", "GIT_AUTHOR_EMAIL": "salih@local",
            "GIT_COMMITTER_NAME": "salih", "GIT_COMMITTER_EMAIL": "salih@local"})

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if r.stdout: print(r.stdout.strip())
    if r.stderr: print(r.stderr.strip())

for f in ["do_commit.py", "do_push.py", "gpu_tropical_new.py", "test_gpu.py"]:
    p = os.path.join(cwd, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed {f}")

run(["git", "add", "-A"])
run(["git", "status"])
run(["git", "commit", "-m", "Remove temp files"])
run(["git", "push"])
print("DONE")
