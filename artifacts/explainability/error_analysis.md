# Error Diagnostics & Failure Mode Report

- **Total Evaluated Samples:** 19,052
- **Total Errors:** 382 (2.01%)
- **False Positives (FP):** 250 (FPR: 2.60%)
- **False Negatives (FN):** 132 (FNR: 1.40%)

## Error Distribution by Source

| Dataset Source | Samples | FP | FN | Error Rate |
|---|---|---|---|---|
| Assassin | 880 | 23 | 16 | 4.43% |
| CEAS-08 | 5,350 | 93 | 8 | 1.89% |
| Enron | 4,432 | 87 | 17 | 2.35% |
| Ling | 433 | 9 | 1 | 2.31% |
| TREC-07 | 7,957 | 38 | 90 | 1.61% |

## Top False Positives (Legitimate flagged as Phishing)

| Source | Predicted Phish Prob | Subject Preview |
|---|---|---|
| TREC-07 | 0.9628 | [Reform] Increase Your Penis Width (Girth) By upto 20%. |
| TREC-07 | 0.9383 | [Reform] Photoshop, Windows, Office |
| TREC-07 | 0.9378 | [Reform] 0EM Software |
| CEAS-08 | 0.9170 | money |
| TREC-07 | 0.9147 | [Reform] Investition |

## Top False Negatives (Phishing missed by Detector)

| Source | Predicted Phish Prob | Subject Preview |
|---|---|---|
| Assassin | 0.0817 | Re: change of plans |
| TREC-07 | 0.1157 | For the University Rankings released in March 2007, Macleans surveyed 70,000 stu |
| TREC-07 | 0.1311 | FWD: Requested documents |
| CEAS-08 | 0.1440 | hi |
| CEAS-08 | 0.1627 | Error |
