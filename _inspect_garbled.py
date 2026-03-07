from pathlib import Path
text = Path(r"F:\code space of everything\research-idea-test\hot100.md").read_text(encoding="utf-8")
for line in text.splitlines():
    if "竖线高" in line or "原地" in line and "盖" in line:
        if "�" in line or "" in line:
            print(repr(line))
            print([hex(ord(ch)) for ch in line])
