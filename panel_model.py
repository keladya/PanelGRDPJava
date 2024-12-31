# -*- coding: utf-8 -*-
!pip install linearmodels

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import chi2
import statsmodels.formula.api as smf

import pandas as pd

# Load the spreadsheet
file_path = '/content/data_ekonom.xlsx'
excel_data = pd.ExcelFile(file_path)

# Check sheet names to understand the structure of the file
excel_data.sheet_names

"""# Pooled"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# Load the Excel file
file_path = '/content/data_ekonom.xlsx'
xls = pd.ExcelFile(file_path)

# Read the 'pooled-2014' sheet into a pandas DataFrame
df = pd.read_excel(xls, sheet_name='pooled')

# Transformasi logaritma natural untuk variabel independen dan dependen
independent_variables = ['PAD', 'Kerja_Umur15', 'Tabungan', 'Kredit']
for var in independent_variables:
    df[f'ln_{var}'] = np.log(df[var])

# Transformasi log untuk variabel dependen
df['ln_PDRB'] = np.log(df['PDRB'])

# Tentukan variabel dependen dan independen
dependent_variable = df['ln_PDRB']
independent_variable_ln = df[['ln_PAD', 'ln_Kerja_Umur15', 'ln_Tabungan', 'ln_Kredit']]

# Tambahkan konstanta ke variabel independen
independent_variable_ln = sm.add_constant(independent_variable_ln)

# Fit model OLS menggunakan variabel independen yang sudah di-ln
model_OLS_ln = sm.OLS(dependent_variable, independent_variable_ln).fit()

# Tampilkan ringkasan hasil model
print(model_OLS_ln.summary())

"""# FEM

FEM time
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import chi2

# Transformasi logaritma natural untuk variabel independen dan dependen
independent_variables = ['PAD', 'Kerja_Umur15', 'Tabungan', 'Kredit']
for var in independent_variables:
    df[f'ln_{var}'] = np.log(df[var])

# Transformasi log untuk variabel dependen
df['ln_PDRB'] = np.log(df['PDRB'])

# Atur indeks panel data
df = df.reset_index()
df = df.set_index(['Kabupaten', 'Tahun'])

# Formula regresi
formula = "ln_PDRB ~ " + " + ".join([f'ln_{var}' for var in independent_variables]) + " + TimeEffects"

# Estimasi model Fixed Effects (FEM)
model_fem_time = PanelOLS.from_formula(formula, df)
results_fem_time = model_fem_time.fit(cov_type='clustered', cluster_entity=True)

print(results_fem_time)

"""FEM indv"""

# Transformasi logaritma natural untuk variabel independen dan dependen
independent_variables = ['PAD', 'Kerja_Umur15', 'Tabungan', 'Kredit']
for var in independent_variables:
    df[f'ln_{var}'] = np.log(df[var])

df['ln_PDRB'] = np.log(df['PDRB'])

# Atur indeks panel data
df = df.reset_index()
df = df.set_index(['Kabupaten', 'Tahun'])

formula = "ln_PDRB ~ " + " + ".join([f'ln_{var}' for var in independent_variables]) + " + EntityEffects"

model_fem_indv = PanelOLS.from_formula(formula, df)
results_fem_indv = model_fem_indv.fit(cov_type='clustered', cluster_entity=True)

print(results_fem_indv.summary)

"""Dengan detail intercept sebagai berikut"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Define the regression formula using the ln-transformed variables
formula = "ln_PDRB ~ ln_PAD + ln_Kerja_Umur15 + ln_Tabungan + ln_Kredit + C(Kabupaten)"

# Reset the index to make 'Kabupaten' and 'Tahun' columns
df = df.reset_index()

#  Fixed Effects Model (FEM) menggunakan OLS
model_fem = smf.ols(formula=formula, data=df)
results_fem = model_fem.fit()

# Tampilkan ringkasan hasil model
print(results_fem.summary())

"""FEM both"""

# Transformasi logaritma natural untuk variabel independen dan dependen
independent_variables = ['PAD', 'Kerja_Umur15', 'Tabungan', 'Kredit']
for var in independent_variables:
    df[f'ln_{var}'] = np.log(df[var])

# Transformasi log untuk variabel dependen
df['ln_PDRB'] = np.log(df['PDRB'])

# Atur indeks panel data
df = df.reset_index()
df = df.set_index(['Kabupaten', 'Tahun'])

# Formula regresi
formula = "ln_PDRB ~ " + " + ".join([f'ln_{var}' for var in independent_variables]) + " + EntityEffects + TimeEffects"

# Estimasi model Fixed Effects (FEM)
model_fem_both = PanelOLS.from_formula(formula, df)
results_fem_both = model_fem_both.fit(cov_type='clustered', cluster_entity=True)
print(results_fem_both)

# Tampilkan semua atribut yang tersedia dalam objek hasil model
print(dir(results_fem_time))

# Jumlah parameter (k) dan jumlah observasi (n) dari setiap model
k_fem_time = results_fem_time.params.shape[0]  # Jumlah parameter model FEM_Time
k_fem_indv = results_fem_indv.params.shape[0]  # Jumlah parameter model FEM_Entity
k_fem_both = results_fem_both.params.shape[0]  # Jumlah parameter model FEM_Both

n_fem_time = results_fem_time.nobs  # Jumlah observasi model FEM_Time
n_fem_indv = results_fem_indv.nobs  # Jumlah observasi model FEM_Entity
n_fem_both = results_fem_both.nobs  # Jumlah observasi model FEM_Both

# Log-likelihood dari setiap model
loglik_fem_time = results_fem_time.loglik
loglik_fem_indv = results_fem_indv.loglik
loglik_fem_both = results_fem_both.loglik

# Hitung AIC dan BIC secara manual
aic_fem_time = -2 * loglik_fem_time + 2 * k_fem_time
aic_fem_indv = -2 * loglik_fem_indv + 2 * k_fem_indv
aic_fem_both = -2 * loglik_fem_both + 2 * k_fem_both

bic_fem_time = -2 * loglik_fem_time + k_fem_time * np.log(n_fem_time)
bic_fem_indv = -2 * loglik_fem_indv + k_fem_indv * np.log(n_fem_indv)
bic_fem_both = -2 * loglik_fem_both + k_fem_both * np.log(n_fem_both)

# Ringkasan hasil model dalam bentuk tabel
results_comparison = pd.DataFrame({
    "Model": ["FEM_Time", "FEM_Entity", "FEM_Both"],
    "R²_within": [results_fem_time.rsquared_within, results_fem_indv.rsquared_within, results_fem_both.rsquared_within],
    "R²_between": [results_fem_time.rsquared_between, results_fem_indv.rsquared_between, results_fem_both.rsquared_between],
    "R²_overall": [results_fem_time.rsquared_overall, results_fem_indv.rsquared_overall, results_fem_both.rsquared_overall],
    "Log-Likelihood": [loglik_fem_time, loglik_fem_indv, loglik_fem_both],
    "AIC": [aic_fem_time, aic_fem_indv, aic_fem_both],
    "BIC": [bic_fem_time, bic_fem_indv, bic_fem_both]
})

# Tampilkan tabel hasil
print("Perbandingan Model FEM:")
results_comparison

"""* Within: Mengukur variansi dalam individu (fixed effects).
* Between: Mengukur variansi antar individu (cross-sectional).
* Overall: Mengukur variansi total, mencakup within dan between.

Jika R²_within negatif:

1. Efek tetap individu danatau efek tetap waktu mungkin tidak signifikan atau tidak relevan dalam menjelaskan variabel dependen.
2. Model mungkin mengalami overfitting karena terlalu banyak parameter dummy yang ditambahkan.
3. Kombinasi efek tetap individu dan waktu sekaligus mungkin menyebabkan redundansi, terutama jika variabel independen sudah menjelaskan sebagian besar variasi data.

pilih FEM_entity atau model_fem_indv

# REM
"""

formula = "ln_PDRB ~ " + " + ".join([f'ln_{var}' for var in independent_variables]) + " + EntityEffects + TimeEffects"
model_rem_2way = RandomEffects.from_formula(formula, df)
results_rem_2way = model_rem_2way.fit()

print(results_rem_2way)

"""# Comparison fem rem pooled"""

# model_OLS_ln = sm.OLS(dependent_variable, independent_variable_ln).fit()

# Estimasi model Fixed Effects (FEM)
# model_fem_indv = PanelOLS.from_formula(formula, df)
# results_fem_indv = model_fem_indv.fit(cov_type='clustered', cluster_entity=True)

# model_rem_2way = RandomEffects.from_formula(formula, df)
# results_rem_2way = model_rem_2way.fit()

"""1. FEM vs OLS"""

import numpy as np
from scipy.stats import f

# Pastikan multi-index untuk data panel
# df = df.set_index(['Kabupaten', 'Tahun'])

# Jumlah individu (N) dan waktu (T)
N = df.index.get_level_values(0).nunique()  # Banyak individu (Kabupaten)
T = df.index.get_level_values(1).nunique()  # Banyak waktu (Tahun)
NT = len(df)  # Total observasi

# Model OLS (Tanpa Efek Tetap)
X_ols = sm.add_constant(df[['ln_PAD', 'ln_Kerja_Umur15', 'ln_Tabungan', 'ln_Kredit']])
y_ols = df['ln_PDRB']
model_ols = sm.OLS(y_ols, X_ols).fit()
RSS_OLS = np.sum(model_ols.resid ** 2)

# Model Fixed Effects (FEM)
formula_fem = "ln_PDRB ~ ln_PAD + ln_Kerja_Umur15 + ln_Tabungan + ln_Kredit + EntityEffects"
model_fem = PanelOLS.from_formula(formula_fem, df)
results_fem = model_fem.fit()
RSS_FEM = np.sum(results_fem.resids ** 2)

# Hitung F-Statistic
numerator = (RSS_OLS - RSS_FEM) / (N - 1)
denominator = RSS_FEM / (NT - N - len(results_fem.params))
F_statistic = numerator / denominator

# Hitung p-value dari distribusi F
df1 = N - 1  # Derajat bebas pembilang
df2 = NT - N - len(results_fem.params)  # Derajat bebas penyebut
p_value = 1 - f.cdf(F_statistic, df1, df2)

# Tampilkan hasil uji F
print("\nHasil Uji F untuk FEM vs OLS:")
print(f"F-statistic: {F_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretasi hasil
if p_value < 0.05:
    print("H0 ditolak: Model Fixed Effects lebih baik dibandingkan OLS.")
else:
    print("H0 tidak ditolak: Model OLS lebih baik atau sama dengan Fixed Effects.")

"""2. REM vs OLS"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import RandomEffects
from linearmodels.panel import PanelOLS
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import RandomEffects
from linearmodels.panel import PanelOLS
from scipy.stats import chi2

from scipy.stats import chi2

# Ambil R-squared dari model OLS_ln (OLS dengan transformasi log)
r_squared_ols_ln = model_OLS_ln.rsquared

# Ambil R-squared dari model Random Effects (REM 2-way)
r_squared_rem_2way = results_rem_2way.rsquared

# Hitung Honda Test Statistic
honda_statistic = (r_squared_rem_2way - r_squared_ols_ln) / (1 - r_squared_ols_ln)

# Hitung p-value dari distribusi chi-square dengan 1 derajat kebebasan
p_value_honda = 1 - chi2.cdf(honda_statistic, df=1)

# Tampilkan hasil uji Honda
print(f"\nHonda Test Statistic: {honda_statistic:.4f}")
print(f"P-Value: {p_value_honda:.4f}")

# Interpretasi hasil
if p_value_honda < 0.05:
    print("Model REM lebih baik dibandingkan OLS berdasarkan uji Honda.")
else:
    print("Model OLS mungkin sudah cukup.")

"""3. REM vs FEM"""

import numpy as np
from scipy.stats import chi2
from numpy.linalg import inv

# Estimasi model Random Effects (REM 2-way)
formula = "ln_PDRB ~ ln_PAD + ln_Kerja_Umur15 + ln_Tabungan + ln_Kredit + EntityEffects + TimeEffects"
model_rem_2way = RandomEffects.from_formula(formula, df)
results_rem_2way = model_rem_2way.fit()

# Estimasi model Fixed Effects (FEM)
model_fem = PanelOLS.from_formula("ln_PDRB ~ ln_PAD + ln_Kerja_Umur15 + ln_Tabungan + ln_Kredit + EntityEffects", df)
results_fem_indv = model_fem_indv.fit(cov_type='clustered', cluster_entity=True)

# Koefisien dan matriks kovarians untuk FEM dan REM
b_FEM = results_fem.params
b_REM = results_rem_2way.params

cov_b_FEM = results_fem_indv.cov
cov_b_REM = results_rem_2way.cov

# Perhitungan statistik uji Hausman
diff_b = b_FEM - b_REM
cov_diff = cov_b_FEM + cov_b_REM

# Periksa apakah matriks kovarian dapat diinvers
try:
    hausman_stat = np.dot(diff_b.T, inv(cov_diff)).dot(diff_b)
    df_hausman = len(b_FEM)  # Jumlah parameter
    p_value_hausman = 1 - chi2.cdf(hausman_stat, df_hausman)

    # Tampilkan hasil uji Hausman
    print(f"\nHausman Test Statistic: {hausman_stat:.4f}")
    print(f"P-Value: {p_value_hausman:.4f}")

    # Interpretasi hasil
    if p_value_hausman < 0.05:
        print("Model FEM lebih tepat dibandingkan REM berdasarkan uji Hausman.")
    else:
        print("Model REM lebih efisien dibandingkan FEM berdasarkan uji Hausman.")

except np.linalg.LinAlgError:
    print("Matriks kovarian tidak dapat diinvers. Uji Hausman gagal dilakukan.")

"""Dipilih FEM

# Asumsi Klasik FEM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Pastikan model sudah dijalankan
model_fem = PanelOLS.from_formula("ln_PDRB ~ ln_PAD + ln_Kerja_Umur15 + ln_Tabungan + ln_Kredit + EntityEffects", df)
results_fem = model_fem.fit(cov_type='clustered', cluster_entity=True)

# Ambil residual dari model FEM
residuals = results_fem.resids

# 1. UJI NORMALITAS RESIDUAL
print("=== UJI NORMALITAS RESIDUAL ===")
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Test Statistic: {stat:.4f}")

# 2. UJI AUTOKORELASI RESIDUAL (Durbin-Watson Test)
print("\n=== UJI AUTOKORELASI RESIDUAL ===")
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_statistic:.4f}")
if 1.5 <= dw_statistic <= 2.5:
    print("Tidak ada autokorelasi residual.")
else:
    print("Ada indikasi autokorelasi residual.")

# 3. UJI HETEROSKEDASTISITAS (Breusch-Pagan Test)
print("\n=== UJI HETEROSKEDASTISITAS ===")
import statsmodels.api as sm

# Sort the data by the predicted values
residuals = results_fem_indv.resids
# Access dependent variable values using .dataframe attribute instead of .values
dependent_variable = results_fem_indv.model.dependent.dataframe.values
sorted_data = sorted(zip(results_fem_indv.fitted_values, residuals), key=lambda x: x[0])
fitted_values_sorted, residuals_sorted = zip(*sorted_data)

# Split the data into two halves
split_point = len(residuals_sorted) // 2
residuals_part1 = np.array(residuals_sorted[:split_point])
residuals_part2 = np.array(residuals_sorted[split_point:])

# Calculate the sum of squared residuals for each part
rss1 = np.sum(residuals_part1**2)
rss2 = np.sum(residuals_part2**2)

# Calculate the Goldfeld-Quandt statistic
gq_statistic = rss1 / rss2

# Get the degrees of freedom for each part
df1 = len(residuals_part1) - 1
df2 = len(residuals_part2) - 1

from scipy.stats import f  # Import f for F-distribution
p_value = 1 - f.cdf(gq_statistic, df1, df2)

# Print the results
print("\nGoldfeld-Quandt Test Results:")
print(f"GQ Statistic: {gq_statistic:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No heteroscedasticity is detected.")

# 4. UJI MULTIKOLINEARITAS (VIF)
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Variabel independen dengan transformasi logaritma natural
independent_variables_ln = [f'ln_{var}' for var in ['PAD', 'Kerja_Umur15', 'Tabungan', 'Kredit']]

# Ambil data hanya untuk variabel independen
X_vif = df[independent_variables_ln]

# Tambahkan konstanta ke data independen untuk perhitungan VIF
X_vif_with_const = sm.add_constant(X_vif)

# Hitung VIF untuk setiap variabel
vif = pd.DataFrame()
vif['Variable'] = X_vif_with_const.columns
vif['VIF'] = [variance_inflation_factor(X_vif_with_const.values, i) for i in range(X_vif_with_const.shape[1])]

# Visualisasi Residual untuk Normalitas
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title("Histogram Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

print("\n=== UJI MULTIKOLINEARITAS ===")
print("Hasil Uji VIF untuk Variabel Independen:")
vif
