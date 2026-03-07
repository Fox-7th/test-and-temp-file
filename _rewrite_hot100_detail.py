import io
import re
import tokenize
from pathlib import Path

path = Path(r"F:\code space of everything\research-idea-test\hot100.md")
text = path.read_text(encoding="utf-8")
text = text.replace("\ufffd", "")  # remove replacement chars
lines = text.splitlines()

new_lines = []
idx = 0
while idx < len(lines):
    line = lines[idx]
    if line.strip().startswith("题目："):
        original = line.strip()[3:]
        # look ahead to see if we already have the inserted lines
        if idx + 3 < len(lines) and lines[idx + 1].strip().startswith("直白版：") and lines[idx + 2].strip().startswith("输入：") and lines[idx + 3].strip().startswith("输出："):
            # compute better input/output description
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

text = "\n".join(new_lines) + "\n"

# Fix numeric 'text' variable in code blocks
code_fence = "```python"
end_fence = "```"
parts = text.split(code_fence)
if len(parts) > 1:
    out = [parts[0]]
    for part in parts[1:]:
        if end_fence not in part:
            out.append(code_fence + part)
            continue
        code, rest = part.split(end_fence, 1)
        block = code
        if "text" in block and re.search(r"\btext\s*[<>=]=?\s*[-\d]", block):
            # replace token-wise text -> sum_value
            tokens = tokenize.generate_tokens(io.StringIO(block).readline)
            new_tokens = []
            for tok in tokens:
                if tok.type == tokenize.NAME and tok.string == "text":
                    tok = tokenize.TokenInfo(tok.type, "sum_value", tok.start, tok.end, tok.line)
                new_tokens.append(tok)
            block = tokenize.untokenize(new_tokens)
        out.append(code_fence + block + end_fence + rest)
    text = "".join(out)

path.write_text(text, encoding="utf-8")
print("Updated hot100.md")
