Held-out validation, test-first.

- Rerun the passing heads' interventions on the held-out seeds from the
  experiment file (never used to choose heads); a head is "confirmed"
  when it passes there too (D13), "tested" when it passed only on the
  finding seeds.
- X_eTaL report: per head, finding vs held-out effect and verdict.
- Tests: the held-out seeds are disjoint from the finding seeds (checked
  from the file); the verdict follows the rule exactly.
