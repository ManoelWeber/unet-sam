import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import unicodedata
import re


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_CSV = r"C:\Projetos\SAM_V2\metricas\metricas_unet.csv"
PASTA_SAIDA = Path(r"C:\Projetos\SAM_V2\resultados\boxplots")

LIMITE_VB = 31  # limite usado para separar desgaste pequeno e maior

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar_nome_coluna(nome):
    """
    Padroniza o nome das colunas para evitar erros por acentos,
    espaços duplicados, símbolos ou pequenas diferenças de escrita.
    """
    nome = str(nome).strip()
    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(c for c in nome if not unicodedata.combining(c))
    nome = nome.lower()
    nome = nome.replace("%", "")
    nome = nome.replace("(", "")
    nome = nome.replace(")", "")
    nome = nome.replace("-", "_")
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    nome = nome.strip("_")
    return nome


def ler_csv(caminho):
    """
    Tenta ler o CSV considerando os formatos mais comuns:
    - separador ponto e vírgula;
    - separador vírgula;
    - decimal com ponto;
    - decimal com vírgula.
    """
    tentativas = [
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": "."},
        {"sep": ",", "decimal": ","},
    ]

    for config in tentativas:
        try:
            df = pd.read_csv(caminho, **config)
            if df.shape[1] > 1:
                print(f"CSV lido com sep='{config['sep']}' e decimal='{config['decimal']}'")
                return df
        except Exception:
            pass

    raise ValueError("Não foi possível ler o CSV. Verifique o separador e o formato do arquivo.")


def converter_numero(serie):
    """
    Converte valores para número, aceitando decimal com vírgula ou ponto.
    """
    return pd.to_numeric(
        serie.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce"
    )


def salvar_boxplot(dados, rotulos, titulo, eixo_y, nome_arquivo, figsize=(9, 5)):
    """
    Gera e salva um boxplot em PNG com 300 dpi.
    """
    plt.figure(figsize=figsize)
    plt.boxplot(dados, labels=rotulos, showmeans=True)

    plt.title(titulo)
    plt.ylabel(eixo_y)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    caminho_saida = PASTA_SAIDA / nome_arquivo
    plt.savefig(caminho_saida, dpi=300)
    plt.close()

    print(f"Arquivo salvo: {caminho_saida}")


def verificar_colunas(df, colunas):
    """
    Verifica se todas as colunas necessárias existem no DataFrame.
    """
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise KeyError(
            "\nAs seguintes colunas não foram encontradas:\n"
            f"{faltantes}\n\n"
            "Colunas disponíveis no arquivo:\n"
            f"{df.columns.tolist()}"
        )


# ============================================================
# LEITURA E PREPARAÇÃO DOS DADOS
# ============================================================

df = ler_csv(ARQUIVO_CSV)

print("\nColunas originais encontradas no CSV:")
print(df.columns.tolist())

# Normaliza os nomes das colunas
df.columns = [normalizar_nome_coluna(col) for col in df.columns]

print("\nColunas normalizadas:")
print(df.columns.tolist())


# Remove linhas de média e desvio padrão, caso estejam no CSV
termos_resumo = ["MEDIA", "MÉDIA", "DESVIO PADRAO", "DESVIO PADRÃO"]

for coluna_texto in ["dataset", "nome_da_imagem"]:
    if coluna_texto in df.columns:
        df = df[
            ~df[coluna_texto]
            .astype(str)
            .str.upper()
            .isin(termos_resumo)
        ]

# Remove linhas sem nome de imagem
if "nome_da_imagem" in df.columns:
    df = df[df["nome_da_imagem"].notna()]

# Colunas necessárias para os gráficos
colunas_necessarias = [
    "nome_da_imagem",
    "vb_manual_px",
    "vb_hibrido_px",
    "dif_vb_hibrido_px",
    "dif_vb_hibrido_abs",
    "iou_hibrido",
    "precision_hibrido",
    "recall_hibrido",
]

verificar_colunas(df, colunas_necessarias)

# Converte colunas numéricas
colunas_numericas = [
    "vb_manual_px",
    "vb_hibrido_px",
    "dif_vb_hibrido_px",
    "dif_vb_hibrido_abs",
    "iou_hibrido",
    "precision_hibrido",
    "recall_hibrido",
]

for coluna in colunas_numericas:
    df[coluna] = converter_numero(df[coluna])

# Remove linhas sem métricas válidas
df = df.dropna(subset=["iou_hibrido", "precision_hibrido", "recall_hibrido", "vb_manual_px"])

# Cria grupo por magnitude do VB manual
df["grupo_vb"] = df["vb_manual_px"].apply(
    lambda x: f"VB ≤ {LIMITE_VB} px" if x <= LIMITE_VB else f"VB > {LIMITE_VB} px"
)

# Calcula diferença absoluta de VB em pixels
df["dif_vb_abs_px"] = df["dif_vb_hibrido_px"].abs()


# ============================================================
# BOXPLOT PRINCIPAL - IoU, PRECISÃO E RECALL
# ============================================================

metricas_principais = [
    "iou_hibrido",
    "precision_hibrido",
    "recall_hibrido",
]

dados_metricas = [df[coluna].dropna() for coluna in metricas_principais]
rotulos_metricas = ["IoU", "Precisão", "Recall"]

salvar_boxplot(
    dados=dados_metricas,
    rotulos=rotulos_metricas,
    titulo="Distribuição das métricas IoU, precisão e recall do pipeline híbrido",
    eixo_y="Valor da métrica (%)",
    nome_arquivo="figura_13_boxplot_iou_precisao_recall_hibrido.png"
)


# ============================================================
# BOXPLOT AUXILIAR - DIFERENÇA PERCENTUAL ABSOLUTA DE VB POR GRUPO
# ============================================================

grupo_menor = f"VB ≤ {LIMITE_VB} px"
grupo_maior = f"VB > {LIMITE_VB} px"

dados_dif_vb_percentual = [
    df.loc[df["grupo_vb"] == grupo_menor, "dif_vb_hibrido_abs"].dropna(),
    df.loc[df["grupo_vb"] == grupo_maior, "dif_vb_hibrido_abs"].dropna(),
]

salvar_boxplot(
    dados=dados_dif_vb_percentual,
    rotulos=[grupo_menor, grupo_maior],
    titulo="Diferença percentual absoluta de VB por magnitude do desgaste",
    eixo_y="Diferença percentual absoluta de VB (%)",
    nome_arquivo="boxplot_dif_vb_percentual_por_grupo_31px.png"
)


# ============================================================
# BOXPLOT AUXILIAR - DIFERENÇA ABSOLUTA DE VB EM PIXELS POR GRUPO
# ============================================================

dados_dif_vb_px = [
    df.loc[df["grupo_vb"] == grupo_menor, "dif_vb_abs_px"].dropna(),
    df.loc[df["grupo_vb"] == grupo_maior, "dif_vb_abs_px"].dropna(),
]

salvar_boxplot(
    dados=dados_dif_vb_px,
    rotulos=[grupo_menor, grupo_maior],
    titulo="Diferença absoluta de VB por magnitude do desgaste",
    eixo_y="Diferença absoluta de VB (px)",
    nome_arquivo="boxplot_dif_vb_px_por_grupo_31px.png"
)


# ============================================================
# BOXPLOT AUXILIAR - IoU, PRECISÃO E RECALL POR GRUPO DE VB
# ============================================================

dados_metricas_por_grupo = [
    df.loc[df["grupo_vb"] == grupo_menor, "iou_hibrido"].dropna(),
    df.loc[df["grupo_vb"] == grupo_maior, "iou_hibrido"].dropna(),
    df.loc[df["grupo_vb"] == grupo_menor, "precision_hibrido"].dropna(),
    df.loc[df["grupo_vb"] == grupo_maior, "precision_hibrido"].dropna(),
    df.loc[df["grupo_vb"] == grupo_menor, "recall_hibrido"].dropna(),
    df.loc[df["grupo_vb"] == grupo_maior, "recall_hibrido"].dropna(),
]

rotulos_metricas_por_grupo = [
    f"IoU\n{grupo_menor}",
    f"IoU\n{grupo_maior}",
    f"Precisão\n{grupo_menor}",
    f"Precisão\n{grupo_maior}",
    f"Recall\n{grupo_menor}",
    f"Recall\n{grupo_maior}",
]

salvar_boxplot(
    dados=dados_metricas_por_grupo,
    rotulos=rotulos_metricas_por_grupo,
    titulo="IoU, precisão e recall por magnitude do desgaste",
    eixo_y="Valor da métrica (%)",
    nome_arquivo="boxplot_iou_precisao_recall_por_grupo_vb_31px.png",
    figsize=(13, 5)
)


# ============================================================
# RESUMOS ESTATÍSTICOS
# ============================================================

resumo_metricas = df[
    [
        "iou_hibrido",
        "precision_hibrido",
        "recall_hibrido",
        "dif_vb_hibrido_abs",
        "dif_vb_abs_px",
        "vb_manual_px",
    ]
].describe()

resumo_por_grupo = df.groupby("grupo_vb")[
    [
        "iou_hibrido",
        "precision_hibrido",
        "recall_hibrido",
        "dif_vb_hibrido_abs",
        "dif_vb_abs_px",
        "vb_manual_px",
    ]
].agg(["count", "mean", "std", "min", "median", "max"])

# Salva os resumos em CSV
resumo_metricas.to_csv(PASTA_SAIDA / "resumo_metricas_hibrido.csv", sep=";", decimal=",")
resumo_por_grupo.to_csv(PASTA_SAIDA / "resumo_metricas_por_grupo_vb_31px.csv", sep=";", decimal=",")

# Mostra no terminal
print("\nResumo das métricas principais:")
print(resumo_metricas)

print("\nResumo por grupo de VB:")
print(resumo_por_grupo)

print("\nQuantidade de imagens por grupo de VB:")
print(df["grupo_vb"].value_counts())

print("\nMaiores diferenças percentuais absolutas de VB:")
print(
    df[
        [
            "nome_da_imagem",
            "vb_manual_px",
            "vb_hibrido_px",
            "dif_vb_hibrido_px",
            "dif_vb_hibrido_abs",
            "grupo_vb",
        ]
    ]
    .sort_values(by="dif_vb_hibrido_abs", ascending=False)
    .head(10)
)

print("\nBoxplots finalizados.")
print(f"Arquivos salvos em: {PASTA_SAIDA}")