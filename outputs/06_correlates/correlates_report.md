# Correlates Report

## Completion
- Status: fully completed.

## Method
- Unit of analysis: school-level 2022 profiles built from student, school, and teacher data.
- Methods used: weighted descriptive aggregation, bivariate correlations, Elastic Net for standardized coefficients, and shallow random-forest permutation importance as a secondary check.
- Interpretation rule: all findings are correlational and observational.

## Strongest correlates by outcome
### school_mean_math
- `school_mean_escs` (composition) coef=34.276, perm_imp=0.741, corr=0.738
- `student_math_selfefficacy_mean` (student_attitudes) coef=28.718, perm_imp=0.181, corr=0.594
- `student_math_reasoning_selfefficacy_mean` (student_attitudes) coef=-10.482, perm_imp=0.007, corr=0.152
- `student_skipping_mean` (attendance) coef=-10.077, perm_imp=0.052, corr=-0.309
- `student_career_info_mean` (aspirations) coef=-8.902, perm_imp=0.012, corr=-0.175
- `student_tardy_mean` (attendance) coef=-8.407, perm_imp=0.010, corr=-0.241
- `student_school_digital_internet_devices_mean` (digital_access) coef=6.737, perm_imp=0.003, corr=0.459
- `student_device_problem_mean` (digital_access) coef=-6.157, perm_imp=0.005, corr=-0.359
### school_risk_score
- `student_homework_mean` (learning_time) coef=-2.529, perm_imp=0.124, corr=-0.345
- `student_math_reasoning_selfefficacy_mean` (student_attitudes) coef=-1.892, perm_imp=0.000, corr=-0.191
- `school_mean_escs` (composition) coef=1.682, perm_imp=0.003, corr=0.234
- `student_skipping_mean` (attendance) coef=-1.605, perm_imp=0.087, corr=-0.224
- `student_ict_regulation_mean` (digital) coef=-1.522, perm_imp=0.449, corr=-0.295
- `student_career_info_mean` (aspirations) coef=-1.389, perm_imp=0.006, corr=-0.172
- `student_expected_edu_mean` (aspirations) coef=-1.351, perm_imp=0.002, corr=-0.216
- `RATCMP2` (digital_access) coef=1.219, perm_imp=0.011, corr=0.264
### within_school_gap
- `school_mean_escs` (composition) coef=6.345, perm_imp=0.153, corr=0.253
- `student_homework_mean` (learning_time) coef=-3.129, perm_imp=0.025, corr=-0.186
- `student_belong_mean` (student_attitudes) coef=2.696, perm_imp=0.013, corr=0.154
- `SCHAUTO` (school_governance) coef=-2.320, perm_imp=0.006, corr=0.017
- `PROPSUPP` (support_staff) coef=1.940, perm_imp=0.003, corr=0.139
- `SC201Q01JA` (leadership) coef=1.820, perm_imp=0.000, corr=0.031
- `NEGSCLIM` (school_climate) coef=1.722, perm_imp=0.006, corr=0.079
- `student_digital_efficacy_mean` (digital) coef=1.704, perm_imp=0.001, corr=0.155

## Caution
- These rankings indicate which observed features co-move with risk, within-school inequality, or average performance after partial regularisation.
- They do not establish why the gaps exist and they are not suitable for claiming causal intervention effects.