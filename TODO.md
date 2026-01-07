# TODO

## 🚀 Core Pipeline (ASAP)
**Priority: High** | **Status: In Progress**

- [ ] Simple CLI commands: streamline common workflows (ingest → rank → schedule/report) while preserving advanced options

---

## 📊 Excel/Report Improvements
**Priority: Medium** | **Status: Complete**

- [x] Add rank column to Excel output
- [x] Dashboard grouped by games
- [x] Totals/summary column on dashboard
- [x] Modular reporting separated from core model math

---

## 🧮 Model Enhancements
**Priority: Medium** | **Status: Planned**

- [ ] Accuracy evaluation framework (consistent metrics + reports across models)
- [ ] Probability calibration curves and summary tables
- [ ] Per-sport home-advantage estimation with confidence intervals


---

## 💰 Features
**Priority: Low** | **Status: Not Started**

- [ ] Betting utilities: pricing (e.g., bet-to-win calculator)

---

## 📝 Documentation & Cleanup
**Priority: Low** | **Status: In Progress**

- [ ] Legacy code review: confirm whether legacy "import sports reference" path is still needed

---

## ✅ Completed
_Move finished items here to track progress_

- [x] Daily data pipeline: consistent rankings and matchup projections
- [x] Excel output: clean worksheets with projected matchups
- [x] Idempotency: repeated runs produce stable outputs
- [x] Home-court advantage integrated into Bradley-Terry
- [x] Margin of victory incorporated into models
- [x] Additional models added (Elo, GSSD, TOOR, etc.)
- [x] Multi-model framework groundwork
- [x] CLI reference documented