import subprocess

def build_legion_info():
    try:
        long_version = subprocess.run(["git", "describe", "--tags"], capture_output=True).stdout.decode().strip()
    except Exception:
        long_version = "<unknown>"

    with open("legion_info.py", "w") as f:
        f.write(f"__version__ = {long_version!r}\n")


if __name__ == '__main__':
    build_legion_info()