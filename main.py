import numpy as np
import pandas as pd

def Normis(*args, X_min = None, X_max = None):
    X = np.array(args)
    X_norm = (X - X_min)/(X_max-X_min)
    return X_norm

data = np.array([
    [3, 10, False],
    [11, 9.8, False],
    [7, 9, True],
    [8.5, 8.9, True],
    [8.3, 8, True],
    [6.3, 6.4, False],
    [12, 6, False],
    [7.1, 6, True],
    [4, 10.1, False],
    [6, 7, False],
    [15, 13, False],
    [6.7, 9.5, True],
    [7.5, 9.8, True]
])

df = pd.DataFrame(data, columns=["Kisl", "Zhest", "Prigodn"])
#print(df)
df["Kisl"] = pd.to_numeric(df["Kisl"], errors='coerce')
df["Zhest"] = pd.to_numeric(df["Zhest"], errors='coerce')
norm_Kisl = Normis(*df.Kisl, X_min = df["Kisl"].min(), X_max = df["Kisl"].max())  # Нормированные значения для кислотности
norm_Zhest = Normis(*df.Zhest, X_min = df["Zhest"].min(), X_max = df["Zhest"].max()) # Нормированные значения для жесткости
print(df["Kisl"].min())
#print(norm_Kisl)
#print(norm_Zhest)

df_norm = pd.DataFrame({
    "Kisl_norm": norm_Kisl,
    "Zhest_norm": norm_Zhest
})

#print(df_norm)
result = pd.concat([df_norm, df["Prigodn"]], axis = 1)
print(result)


def calculate(Stolb_1, Stolb_2, X_2, Y_2):
    X_1 = np.array(Stolb_1)
    Y_1 = np.array(Stolb_2)
    Evclid = (((X_2 - X_1)**2) + ((Y_2 - Y_1)**2))**1/2
    return Evclid


df['Distance'] = calculate(result['Kisl_norm'], result['Zhest_norm'], 6, 12)

res = pd.concat([result, df['Distance']], axis = 1)
print(res)

nearest_indices = df.nsmallest(5, 'Distance').index

# Предсказание классов на основе ближайших соседей
nearest_classes = df.loc[nearest_indices, 'Prigodn']
predicted_class = nearest_classes.mode()[0]  # Получаем наиболее частый класс

print("Предсказанный класс на основе 5 ближайших соседей:", predicted_class)