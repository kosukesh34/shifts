!pip install pulp
import pulp
import pandas as pd

# シフト（曜日と時限）の設定
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
periods = ["2", "3", "4", "5"]
shifts = [d + p for d in days for p in periods]

# 各人の希望シフトとグループ情報
persons_data = {
    "A": {"preferences": ["Mon2", "Mon5", "Thu4", "Fri2", "Fri3"], "group": "G1"},
    "B": {"preferences": ["Mon2", "Tue3", "Wed4", "Thu3", "Fri5"], "group": "G1"},
    "C": {"preferences": ["Tue2", "Tue5", "Wed3", "Thu2", "Fri4"]},
    "D": {"preferences": ["Mon3", "Tue2", "Thu5", "Fri3", "Fri4"]},
    "E": {"preferences": ["Mon4", "Tue4", "Wed2", "Fri2", "Fri5"]},
    "F": {"preferences": ["Tue3", "Wed5", "Thu3", "Fri2", "Mon2"]},
    "G": {"preferences": ["Wed2", "Wed3", "Thu2", "Thu5", "Fri3"]},
    "H": {"preferences": ["Mon5", "Tue2", "Tue4", "Fri3", "Fri4"], "group": "G2"},
    "I": {"preferences": ["Tue5", "Wed4", "Thu3", "Fri2", "Mon4"], "group": "G2"},
    "J": {"preferences": ["Wed3", "Thu2", "Fri5", "Mon3", "Tue4"]},
    "K": {"preferences": ["Mon2", "Wed2", "Fri2", "Tue5", "Thu3"]},
    "L": {"preferences": ["Tue2", "Tue3", "Wed3", "Thu2", "Fri2"]},
    "M": {"preferences": ["Mon3", "Tue5", "Wed4", "Thu4", "Fri5"]},
    "N": {"preferences": ["Mon4", "Tue2", "Wed5", "Thu3", "Fri2"]},
    "O": {"preferences": ["Tue4", "Wed2", "Thu2", "Fri3", "Mon5"], "group": "G3"},
    "P": {"preferences": ["Mon2", "Tue3", "Wed3", "Thu4", "Fri4"], "group": "G3"},
    "Q": {"preferences": ["Mon5", "Tue5", "Wed2", "Thu2", "Fri2"], "group": "G3"},
    "R": {"preferences": ["Tue2", "Wed2", "Thu5", "Fri3", "Mon4"]},
    "S": {"preferences": ["Mon3", "Tue4", "Wed4", "Thu3", "Fri2"]},
    "T": {"preferences": ["Mon2", "Tue2", "Wed3", "Thu4", "Fri5"]}
}

persons = list(persons_data.keys())

# 希望順位に応じたスコア設定（1位:5, 2位:4, ... 5位:1）
pref_score = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
score = {p: {s: 0 for s in shifts} for p in persons}
for p in persons:
    prefs = persons_data[p]["preferences"]
    for rank, s in enumerate(prefs, start=1):
        score[p][s] = pref_score[rank]

# 最適化問題の定義
prob = pulp.LpProblem("Duty_Scheduling", pulp.LpMaximize)

# 変数定義
# x[p][s]: 人pがシフトsに割り当てられるかどうか（0または1）
x = pulp.LpVariable.dicts("assign", (persons, shifts), cat="Binary")
# y[s]: シフトsが使用されているかどうか（0または1）
y = pulp.LpVariable.dicts("used", shifts, cat="Binary")

# 目的関数：全体の希望スコアの最大化
prob += pulp.lpSum(score[p][s] * x[p][s] for p in persons for s in shifts)

# 制約条件1: 各人は必ず1つのシフトに割り当て
for p in persons:
    prob += pulp.lpSum(x[p][s] for s in shifts) == 1

# 制約条件2: 各シフトは、使用される場合、最低2人以上、最大5人まで割り当て
for s in shifts:
    prob += pulp.lpSum(x[p][s] for p in persons) >= 2 * y[s]
    prob += pulp.lpSum(x[p][s] for p in persons) <= 5 * y[s]

# 制約条件3: 同じグループの人は同じシフトに割り当て
groups = {}
for p in persons:
    if "group" in persons_data[p]:
        group_id = persons_data[p]["group"]
        groups.setdefault(group_id, []).append(p)
for group_id, group_persons in groups.items():
    for s in shifts:
        for i in range(len(group_persons) - 1):
            p1 = group_persons[i]
            p2 = group_persons[i + 1]
            prob += x[p1][s] - x[p2][s] == 0

# 最適化問題の解決
prob.solve()

# シフト割当結果の出力
data = []
for s in shifts:
    assigned = [p for p in persons if pulp.value(x[p][s]) > 0.5]
    if assigned:
        data.append({"Time": s, "Persons": ", ".join(assigned)})
df = pd.DataFrame(data)
print("【シフト割当】")
print(df.to_string(index=False))

# 希望シフトに含まれていない割当の確認
non_preferred = []
for p in persons:
    # 各人は必ず1つのシフトに割り当てられているので、該当するシフトを取得
    assigned_shifts = [s for s in shifts if pulp.value(x[p][s]) > 0.5]
    if assigned_shifts:
        assigned_shift = assigned_shifts[0]
        # 割り当てられたシフトが希望リストにない場合、その人を記録
        if assigned_shift not in persons_data[p]["preferences"]:
            non_preferred.append({
                "Person": p,
                "Assigned": assigned_shift,
                "Preferences": ", ".join(persons_data[p]["preferences"])
            })

if non_preferred:
    df_non_preferred = pd.DataFrame(non_preferred)
    print("\n【希望通りでない人の一覧】")
    print(df_non_preferred.to_string(index=False))
else:
    print("\n全員の割当が希望通りです。")
