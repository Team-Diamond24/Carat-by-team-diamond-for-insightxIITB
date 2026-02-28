"""One-shot script to remove orphaned lines from engine.py"""
import os

filepath = os.path.join(os.path.dirname(__file__), "engine.py")

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Original file has {len(lines)} lines")

# Find the clean return statement and the next def
new_lines = []
skip = False
for i, line in enumerate(lines):
    lineno = i + 1  # 1-indexed

    # Start skipping after we see the corrected return on line ~164
    if "return question, corrections, suggestions" in line and not skip:
        new_lines.append("    return question, corrections, suggestions\n")
        skip = True
        continue

    # Stop skipping when we hit the next function definition
    if skip and line.startswith("def "):
        skip = False
        new_lines.append("\n\n")
        new_lines.append(line)
        continue

    if not skip:
        new_lines.append(line)

print(f"New file has {len(new_lines)} lines")
print(f"Removed {len(lines) - len(new_lines)} lines")

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Done! engine.py has been cleaned up.")
