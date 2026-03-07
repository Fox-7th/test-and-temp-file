import re
from pathlib import Path

path = Path(r"F:\code space of everything\research-idea-test\hot100.md")
lines = path.read_text(encoding="utf-8").splitlines()

new_lines = []
idx = 0
while idx < len(lines):
    line = lines[idx]
    if line.strip().startswith("题目："):
        if idx + 3 < len(lines) and lines[idx + 1].strip().startswith("直白版：") and lines[idx + 2].strip().startswith("输入：") and lines[idx + 3].strip().startswith("输出："):
            output_desc = lines[idx + 3].strip()[3:]
            # check if task line already exists
            if idx + 4 < len(lines) and lines[idx + 4].strip().startswith("你要做的事："):
                new_lines.extend(lines[idx:idx + 5])
                idx += 5
                continue
            new_lines.append(line)
            new_lines.append(lines[idx + 1])
            new_lines.append(lines[idx + 2])
            new_lines.append(lines[idx + 3])
            new_lines.append(f"你要做的事：把输入处理成“{output_desc}”并返回。")
            idx += 4
            continue
    new_lines.append(line)
    idx += 1

path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Inserted task lines")
