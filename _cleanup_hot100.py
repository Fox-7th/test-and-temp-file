import re
from pathlib import Path

path = Path(r"F:\code space of everything\research-idea-test\hot100.md")
text = path.read_text(encoding="utf-8")
text = text.replace("\ufffd", "")
text = text.replace("竖线高\n", "竖线高度\n")
text = text.replace("竖线高\r\n", "竖线高度\r\n")
text = text.replace("原地盖", "原地覆盖")

lines = text.splitlines()
new_lines = []
idx = 0
while idx < len(lines):
    line = lines[idx]
    if line.strip().startswith("题目："):
        original = line.strip()[3:]
        if idx + 3 < len(lines) and lines[idx + 1].strip().startswith("直白版：") and lines[idx + 2].strip().startswith("输入：") and lines[idx + 3].strip().startswith("输出："):
            input_desc = None
            m_in = re.search(r"给定([^，。]*)", original)
            if m_in:
                input_desc = m_in.group(1).strip()
            if not input_desc:
                input_desc = "题目中给定的所有输入（如数组/字符串/图/树等）"

            output_desc = None
            m_out = re.search(r"返回([^。；]*)", original)
            if m_out:
                output_desc = m_out.group(1).strip()
            elif "判断" in original or "是否" in original:
                output_desc = "是否满足条件（True/False）"
            elif "求" in original or "最少" in original or "最大" in original or "最小" in original:
                output_desc = "题目要求的数值结果"
            elif "找出" in original or "找到" in original:
                output_desc = "符合条件的结果"
            if not output_desc:
                output_desc = "按题目要求返回结果"

            new_lines.append(line)
            new_lines.append(lines[idx + 1])
            new_lines.append(f"输入：{input_desc}")
            new_lines.append(f"输出：{output_desc}")
            idx += 4
            continue

    new_lines.append(line)
    idx += 1

path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Cleaned hot100.md")
