import numpy as np
from scipy.stats import mannwhitneyu

# Grupo 1: VB manual <= 31 px
# Diferença percentual absoluta de VB (%), em ordem crescente
grupo_menor_igual_31 = np.array([
    3.23,
    4.76,
    7.41,
    14.81,
    16.67,
    25.81,
    42.86,
    50.00
])

# Grupo 2: VB manual > 31 px
# Diferença percentual absoluta de VB (%), em ordem crescente
grupo_maior_31 = np.array([
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    1.32,
    2.44,
    2.70,
    2.82,
    2.86,
    2.90,
    2.99,
    3.45,
    3.85,
    4.00,
    4.65,
    5.26,
    5.41,
    5.41,
    6.00,
    6.25,
    7.32
])

# Teste de Mann-Whitney bicaudal
resultado = mannwhitneyu(
    grupo_menor_igual_31,
    grupo_maior_31,
    alternative="two-sided",
    method="asymptotic"
)

print("Grupo VB <= 31 px")
print("n =", len(grupo_menor_igual_31))
print("média =", np.mean(grupo_menor_igual_31))
print("mediana =", np.median(grupo_menor_igual_31))
print("desvio padrão =", np.std(grupo_menor_igual_31, ddof=1))

print("\nGrupo VB > 31 px")
print("n =", len(grupo_maior_31))
print("média =", np.mean(grupo_maior_31))
print("mediana =", np.median(grupo_maior_31))
print("desvio padrão =", np.std(grupo_maior_31, ddof=1))

print("\nTeste de Mann-Whitney")
print("U =", resultado.statistic)
print("p-valor =", resultado.pvalue)