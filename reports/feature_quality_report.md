# Enterprise Feature Quality Analysis Report

**Project**: BioDCASE2026_Avian_Counting  
**Environment**: development  
**Evaluated Features**: 3  
**Evaluated Samples**: 10  
**Redundancy Index**: **0.3333**  

---

## 1. Quality Overview & Health Checks

| Check | Result | Threshold / Standard | Status |
| :--- | :--- | :--- | :--- |
| **Missing Values** | 0 total | 0 | PASS |
| **Low-Variance Features** | 1 features | 0 | WARNING |
| **Highly Collinear Pairs** | 1 pairs | < 10 | PASS |
| **Feature Redundancy Ratio** | 0.3333 | < 0.20 | WARNING |

---

## 2. Low-Variance & Uninformative Features
- **Flagged Features (Var < 1e-6)**: `zero_var`

---

## 3. High Collinearity & Feature Redundancy (> 0.85 Pearson Correlation)

| Feature 1 | Feature 2 | Absolute Correlation |
| :--- | :--- | :--- |
| `rms_mean` | `rms_std` | **1.0** |


---

## 4. Top Features by Mutual Information (Target Correlation)

| Feature Name | Mutual Information Score |
| :--- | :--- |
| `rms_mean` | **0.8373** |
| `rms_std` | **0.804** |
| `zero_var` | **0.0** |
