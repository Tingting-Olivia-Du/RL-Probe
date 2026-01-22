import json

# 加载两个文件
with open('data/filtered/dpo_results.json', 'r') as f:
    dpo_results = json.load(f)

with open('data/filtered/rlvr_results.json', 'r') as f:
    rlvr_results = json.load(f)

print(f"DPO 结果总数: {len(dpo_results)}")
print(f"RLVR 结果总数: {len(rlvr_results)}")
print()

# 找出 problem id match 的情况
dpo_ids = set(dpo_results.keys())
rlvr_ids = set(rlvr_results.keys())

matched_ids = dpo_ids & rlvr_ids
dpo_only_ids = dpo_ids - rlvr_ids
rlvr_only_ids = rlvr_ids - dpo_ids

print("=== Match 统计 ===")
print(f"两个都有的 ID 数: {len(matched_ids)}")
print(f"只在 DPO 中的 ID 数: {len(dpo_only_ids)}")
print(f"只在 RLVR 中的 ID 数: {len(rlvr_only_ids)}")
print()

# 对于 match 的 ID，统计各种结果组合
match_stats = {
    'both_true': 0,      # DPO=True, RLVR=True
    'both_false': 0,     # DPO=False, RLVR=False
    'dpo_true_rlvr_false': 0,   # DPO=True, RLVR=False
    'dpo_false_rlvr_true': 0,   # DPO=False, RLVR=True
}

match_details = {k: [] for k in match_stats.keys()}

for qid in matched_ids:
    dpo_correct = dpo_results[qid]
    rlvr_correct = rlvr_results[qid]
    
    if dpo_correct and rlvr_correct:
        match_stats['both_true'] += 1
        match_details['both_true'].append(qid)
    elif not dpo_correct and not rlvr_correct:
        match_stats['both_false'] += 1
        match_details['both_false'].append(qid)
    elif dpo_correct and not rlvr_correct:
        match_stats['dpo_true_rlvr_false'] += 1
        match_details['dpo_true_rlvr_false'].append(qid)
    else:  # not dpo_correct and rlvr_correct
        match_stats['dpo_false_rlvr_true'] += 1
        match_details['dpo_false_rlvr_true'].append(qid)

print("=== Match ID 的结果组合 ===")
print(f"DPO✓ RLVR✓ (都对): {match_stats['both_true']}")
print(f"DPO✗ RLVR✗ (都错): {match_stats['both_false']}")
print(f"DPO✓ RLVR✗ (DPO对RLVR错): {match_stats['dpo_true_rlvr_false']}")
print(f"DPO✗ RLVR✓ (DPO错RLVR对): {match_stats['dpo_false_rlvr_true']}")
print()

# 显示差异的例子
if match_details['dpo_true_rlvr_false']:
    print(f"【DPO对RLVR错】的前5个例子:")
    for qid in match_details['dpo_true_rlvr_false'][:5]:
        print(f"  {qid}")
    print()

if match_details['dpo_false_rlvr_true']:
    print(f"【DPO错RLVR对】的前5个例子:")
    for qid in match_details['dpo_false_rlvr_true'][:5]:
        print(f"  {qid}")
    print()

# 保存详细结果
output_file = 'data/filtered/match_problem_analysis.json'
output_data = {
    'summary': {
        'total_dpo': len(dpo_results),
        'total_rlvr': len(rlvr_results),
        'matched': len(matched_ids),
        'dpo_only': len(dpo_only_ids),
        'rlvr_only': len(rlvr_only_ids),
    },
    'matched_stats': match_stats,
    'matched_ids': list(matched_ids),
    'dpo_only_ids': list(dpo_only_ids),
    'rlvr_only_ids': list(rlvr_only_ids),
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ 详细结果已保存到: {output_file}")