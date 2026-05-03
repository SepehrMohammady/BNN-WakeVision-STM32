import re
text = """... (same text) ..."""
with open('sum_cycles.py', 'r') as f:
    text = f.read()

text = text.split('text = """')[1].split('"""')[0]

ops_count = 0
for line in text.strip().split('\n'):
    if "OP Profiler" in line:
        ops_count += 1
print(f"Number of printed ops: {ops_count}")
