import re
from pathlib import Path

path = Path(r"F:\code space of everything\research-idea-test\hot100.md")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()

new_lines = []
idx = 0
while idx < len(lines):
    line = lines[idx]
    # Expand 示例 lines
    if line.strip().startswith("示例：") and "输入：" in line and "输出：" in line:
        content = line.strip()[3:]
        m = re.search(r"输入：(.+?)；输出：(.+)$", content)
        if m:
            inp = m.group(1).strip()
            out = m.group(2).strip()
            new_lines.append("示例：")
            new_lines.append(f"输入：{inp}")
            new_lines.append(f"输出：{out}")
        else:
            new_lines.append(line)
        idx += 1
        continue

    # Expand 题目 lines
    if line.strip().startswith("题目："):
        if idx + 1 < len(lines) and lines[idx + 1].strip().startswith("直白版："):
            new_lines.append(line)
            idx += 1
            continue
        original = line.strip()[3:]
        simple = original
        simple = simple.replace("给定", "有")
        simple = simple.replace("返回", "要返回")
        simple = simple.replace("判断", "要判断")
        simple = simple.replace("求", "要算出")
        simple = simple.replace("找出", "要找到")
        simple = simple.replace("使", "让")
        simple = simple.replace("并", "并且")
        simple = simple.replace("若", "如果")

        output_hint = None
        m = re.search(r"返回([^。；]*)", original)
        if m:
            output_hint = m.group(1).strip()
        elif "判断" in original or "是否" in original:
            output_hint = "是否满足条件（True/False）"
        elif "求" in original or "最少" in original or "最大" in original or "最小" in original:
            output_hint = "题目要求的数值结果"
        elif "找出" in original or "找到" in original:
            output_hint = "符合条件的结果"
        if output_hint:
            output_line = f"输出：{output_hint}"
        else:
            output_line = "输出：按题目要求返回结果"
        input_hint = "题目中给定的所有输入（如数组/字符串/图/树等）"

        new_lines.append(line)
        new_lines.append(f"直白版：{simple}")
        new_lines.append(f"输入：{input_hint}")
        new_lines.append(output_line)
        idx += 1
        continue

    new_lines.append(line)
    idx += 1

path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Updated hot100.md")
