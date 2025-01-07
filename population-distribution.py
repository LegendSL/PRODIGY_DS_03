import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path_population = "API_SP.POP.TOTL_DS2_en_csv_v2_900.csv"
population_data = pd.read_csv(file_path_population, skiprows=4)  # Skipping metadata rows
population_data.head()

year = "2023"
population_2023 = population_data[["Country Name", year]].dropna()

top_countries = population_2023.sort_values(by=year, ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_countries["Country Name"], top_countries[year] / 1e6, color='blue', edgecolor='black')
plt.title(f'Top 10 Countries by Population in {year}', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Population (in millions)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()