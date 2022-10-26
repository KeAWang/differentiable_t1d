# Differentiable T1D Simulator

Implementation of the UVa/Padova 2008 Simulator for Type 1 Diabetes from

> Kovatchev et al., 2009. In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes.

Example PyTorch code: `python uva_padova_torch.py`

Code based on [simglucose](https://github.com/jxx123/simglucose/)

## Glossary of terms in the simulator
### Rate of glucose entering the blood from the stomach (oral glucose subsystem)

| State index | Descriptive name | Units | Description |
| --- | --- | --- | --- |
| 0 | q_sto1 | mg | amount of solid glucose in stomach |
| 1 | q_sto2 | mg | amount of liquid glucose in stomach |
| 2 | q_gut | mg | amount of glucose in the intestine |
| --- | Ra | mg/kg/min | rate of glucose absorption per body weight |
| --- | kmax | 1/min | --- |
| --- | kmin | 1/min | --- |
| --- | kabs | 1/min | --- |
| --- | kgut | 1/min | --- |
| --- | carbs | mg/min | rate of glucose ingested instantaneously |
| --- | f | 1 | one over the fraction of the total ingested glucose; i.e. total / D |

### Glucose kinetics

| State index | Descriptive name | Units | Description |
| --- | --- | --- | --- |
| 3 | G_p | mg/kg | glucose mass in plasma per kg of bodyweight |
| 4 | G_t | mg/kg | glucose mass in tissue per kg of bodyweight |
| --- | EGP | mg/kg/min | endogenous glucose production |
| --- | Uii | mg/kg/min | insulin independent glucose utilization |
| --- | G | mg/dL | plasma glucose concentration |
| 12  | Gs | mg/kg | glucose concentration in the subcutaneous tissue |

### Insulin kinetics

| State index | Descriptive name | Units | Description |
| --- | --- | --- | --- |
| 5 | I_p | pmol/kg | insulin masses in plasma per kg of bodyweight |
| 6 | X | pmol/L | insulin in the interstitial fluid |
| 7 | I_1 | pmol/L | delayed insulin compartment 1 (I tilde in Andy's paper)|
| 8 | I_d | pmol/L | delayed insulin compartment 2 (X_L in Andy's paper)|
| 9 | I_l | pmol/kg | insulin masses in plasma per kg of bodyweight |
| --- | I | pmol/L | plasma insulin concentration |
| --- | insulin | pmol/kg/min | insulin infusion rate per kg of bodyweight |
| --- | V_I | L/kg | distribution volume of insulin |
| 10 | I_sc1 | pmol/kg | nonmonomeric insulin in subcutaneous space |
| 11 | I_sc2 | pmol/kg | monomeric insulin in subcutaneous space |
| --- | Ra_I | pmol/kg | rate of insulin absorption into plasma |
