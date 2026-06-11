# NLA Qwen Team Report Bundle

Date: 2026-05-22

This directory contains the implemented team-report workflow for the local
Qwen2.5-7B layer-20 NLA reproduction and critique.

Files:

- `00_experiment_director_critique.md`: paper-driven critique and assignment brief.
- `01_reproduction_quant_baseline.md`: experimenter 1 report on reproduction and quantitative baseline.
- `02_limitations_failure_modes.md`: experimenter 2 report on limitations and failure modes.
- `03_hidden_state_detection.md`: experimenter 3 report on Plan E hidden-state probes.
- `04_safety_dynamics_position_resolved.md`: experimenter 4 report on Plan C and position-resolved safety dynamics.
- `NLA_Qwen_批判式复现实验报告.md`: final synthesized Chinese report.

Primary evidence paths:

- `/NAS/chennc/NashChennc/.tmp/exp/quant_report.md`
- `/NAS/chennc/NashChennc/.tmp/exp/quant_scores.csv`
- `/NAS/chennc/NashChennc/.tmp/exp/position_resolved/position_report.md`
- `/NAS/chennc/NashChennc/.tmp/exp/position_resolved/position_scores.csv`
- `/NAS/chennc/NashChennc/.tmp/exp/plan_{a,b,c,d,e}/*.decode.log`
