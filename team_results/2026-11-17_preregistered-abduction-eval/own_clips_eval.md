# Own-clips evaluation

_commit: `dedd7c31a1437435d50ffd4ef441fd77f5d25df9` (working tree clean)_

_landmark_space: `world`_

_configs: demo=`config\exercises\shoulder_abduction.json`  reference_relative=derived in this run_

_manifest_hash: `f21c289738ca3e72`  date: 2026-10-31_


## Per-clip rep count

| clip | intended | detected | expected | exact_match | within +/-1 | signed_error |
|---|---|---|---|---|---|---|
| data/recordings/abd_eval_fault_elbow_01.mp4 | fault_elbow | 3 | 3 | True | True | +0 |
| data/recordings/abd_eval_fault_fast_01.mp4 | fault_fast | 3 | 3 | True | True | +0 |
| data/recordings/abd_eval_correct_01.mp4 | correct | 6 | 6 | True | True | +0 |
| data/recordings/abd_eval_fault_lowarm_01.mp4 | fault_lowarm | 3 | 3 | True | True | +0 |
| data/recordings/abd_eval_fault_trunk_02.mp4 | fault_trunk | 3 | 3 | True | True | +0 |

## Average processing speed across eval clips: 31.3 frames/s


## Demo config (absolute)

- false-fail rate (correct reps): 6/6 (100.0%, 95% CI [61.0%, 100.0%])
- which rules fire wrongly on correct reps: {'elbow_straight': 6}
- fault_lowarm (expected rule: peak_angle): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=0/3 (0.0%, 95% CI [0.0%, 56.1%])
- fault_trunk (expected rule: trunk_lean): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=0/3 (0.0%, 95% CI [0.0%, 56.1%])
- fault_elbow (expected rule: elbow_straight): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])
- fault_fast (expected rule: tempo): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=2/3 (66.7%, 95% CI [20.8%, 93.9%])
- insufficient_data rate (all eval reps): 0/18 (0.0%, 95% CI [0.0%, 17.6%])

## Reference_relative config (derived from calibration clips)

- false-fail rate (correct reps): 1/6 (16.7%, 95% CI [3.0%, 56.4%])
- which rules fire wrongly on correct reps: {'tempo': 1}
- fault_lowarm (expected rule: peak_angle): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])
- fault_trunk (expected rule: trunk_lean): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])
- fault_elbow (expected rule: elbow_straight): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=0/3 (0.0%, 95% CI [0.0%, 56.1%])
- fault_fast (expected rule: tempo): any-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])  expected-rule=3/3 (100.0%, 95% CI [43.9%, 100.0%])
- insufficient_data rate (all eval reps): 0/18 (0.0%, 95% CI [0.0%, 17.6%])