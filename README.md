# Unit Commitment with Renewable Curtailment

Educational optimization models for unit commitment in a renewable-rich microgrid, with base and photovoltaic-curtailment formulations.

## Contents

| File | Purpose |
|---|---|
| `unit_commitment.py` | Base mixed-integer unit-commitment formulation with PV, wind, and energy storage. |
| `unit_commitment_with_curtailment.py` | Historical PV-curtailment formulation. |
| `preprocess_kpx_data.py` | Statistics and plots for KPX PV, wind, and load data. |
| `20kW_PV.xlsx`, `80kW_PV.xlsx`, `WT.xlsx`, `Load.xlsx` | Spreadsheet inputs used by the base formulation. |
| `KPX_*.csv` | Historical PV, wind, and load source series. |

## Requirements

- Python 3.8 or later
- `cvxpy`, `numpy`, `pandas`, `matplotlib`, `pulp`, `scipy`, and `openpyxl`
- A mixed-integer solver supported by CVXPY; the base script explicitly requests MOSEK

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python unit_commitment.py
```

The base script expects the four tracked spreadsheet inputs in the repository root. The curtailment script references `200kW_PV.xlsx` and `600kW_PV.xlsx`, which are not currently tracked; it is therefore not a reproducible entry point until those inputs are documented or the script is reconfigured.

## Data notes

KPX-derived input series require source, date, units, and redistribution-permission review before public repackaging. Existing input workbooks are retained unchanged in this documentation branch.

## Cleanup status

This branch normalizes the three root Python filenames while retaining historical input data unchanged. Data-file renaming and structural moves will follow a tested path map; no historical material is removed.

## License and citation

A reuse license and citation guidance will be added after the intended reuse terms are chosen.
