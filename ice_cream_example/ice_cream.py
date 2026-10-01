# 1. The Ice Cream and Shark Attacks (The Confounder Test)
# The Setup: Data shows a strong statistical correlation between ice cream sales at a beach snack bar and 
# the number of shark attacks in the water.
# The Causal Question: Does eating ice cream cause shark attacks?
# The Causal Graph:
# Variable X (Ice Cream Sales) → Variable Y (Shark Attacks)
# Variable Z (Temperature / Sunny Weather) → X and Z → Y [1]
# How to Test It: Feed the data and the known confounder (Z = Temperature) into your Causal AI tool. 
# A correct causal model will determine that the direct effect of ice cream on shark attacks is zero once you adjust for temperature 
# (the common cause). [1]

import pyro
import torch
import numpy as np
import pandas as pd

np.random.seed(1)

n = 500

temperature = np.random.normal(80, 10, n)
ice_cream_sales = (temperature / 2) + np.random.normal(0,2,n)

shark_rate = (temperature -50) / 30
shark_rate = np.clip(shark_rate, 0.05, None)
shark_attacks = np.random.poisson(shark_rate)



df = pd.DataFrame({
    "temperature": temperature,
    "ice_cream_sales": ice_cream_sales,
    "shark_attacks": shark_attacks
}) 

print(df.head(10))

ice_cream_shark_attack_correlation = df["ice_cream_sales"].corr(df["shark_attacks"])
temperature_shark_attack_correlation = df["temperature"].corr(df["shark_attacks"])
ice_cream_temperature_correlation = df["ice_cream_sales"].corr(df["temperature"])
print("The correlation between ice cream sales and shark attacks is:", ice_cream_shark_attack_correlation)
print("The correlation between temperature and shark attacks is:", temperature_shark_attack_correlation)
print("The correlation between ice cream sales and temperature is:", ice_cream_temperature_correlation)



