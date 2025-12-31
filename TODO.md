# TODO

## 🚀 Core Pipeline (ASAP)
**Priority: High** | **Status: Almost Complete**

- [ ] **Simple CLI commands**: Create streamlined commands for common workflows (ingest → calculate → report) while preserving detailed/specific options when needed

---

## 📊 Excel/Report Improvements
**Priority: Medium** | **Status: Not Started**

- [x] **Add rank column**: Include team rankings in Excel output
- [x] **Dashboard grouping**: Group dashboard results by games rather than by model for easier copy/paste of results
- [x] **Totals column**: Add a totals/summary column to the dashboard page
- [x] **Modular reporting**: Separate report generation from mathematical calculations so adding/removing report columns doesn't affect core math (overtime tracking, neutral site tracking, etc. should remain independent)

---

## 🧮 Model Enhancements
**Priority: Medium** | **Status: started**
- [ ] system to test model accuracy
- [ ] 


---

## 💰 Features
**Priority: Low** | **Status: Not Started**

- [ ] **Betting utilities**: Add pricing mechanism (e.g., "bet to win" target calculator)

---

## 📝 Documentation & Cleanup
**Priority: Low** | **Status: In Progress**

- [ ] **Legacy code review**: Verify if "import sports reference" section is still needed or can be removed as outdated

---

## ✅ Completed
_Move finished items here to track progress_

- [x] **Daily data pipeline**: Produce consistent outputs including team power rankings and daily matchups with projections/spreads from the current schedule
- [x] **Excel output**: Generate clean, human-readable Excel worksheets showing up-to-date calendar with projected matchups
- [x] **Idempotency**: Ensure running updates multiple times with no new data returns the same results without duplication
- [x] **Add rank column**: Include team rankings in Excel output
- [x] **Dashboard grouping**: Group dashboard results by games rather than by model for easier copy/paste of results
- [x] **Totals column**: Add a totals/summary column to the dashboard page
- [x] **Modular reporting**: Separate report generation from mathematical calculations so adding/removing report columns doesn't affect core math (overtime tracking, neutral site tracking, etc. should remain independent)
- [x] **Home-court advantage**: Enhance Bradley-Terry model to include home-court advantage factor
- [x] **Margin of victory (MOV)**: Incorporate MOV directly into the model, not just as post-fit scaling
- [x] **Additional models**: Add alternative power ranking models using the same data source
- [x] **Multi-model framework**: Evaluate ssat integration for multi-model comparisons and model reuse
- [x] **CLI reference**: Create comprehensive list of CLI commands and flags (--sport, --season, --input, --db, etc.)
- [x] **Home-court advantage**: Enhance Bradley-Terry model to include home-court advantage factor
- [x] **Margin of victory (MOV)**: Incorporate MOV directly into the model, not just as post-fit scaling
- [x] **Additional models**: Add alternative power ranking models using the same data source
- [x] **Multi-model framework**: Evaluate ssat integration for multi-model comparisons and model reuse