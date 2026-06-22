import os
import cv2
import csv
import numpy as np

# ============================================================
# 1. CONFIGURAÇÕES DOS DIRETÓRIOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pastas de entrada (Onde estão as imagens originais para pegar os nomes)
PASTAS_DADOS = ["train", "valid", "test"]

# Pastas com as máscaras que serão comparadas
DIR_MANUAL = os.path.join(BASE_DIR, "Anotacao Manual")       # Ground Truth (A Base da Verdade)
DIR_SAM_ISOLADO = os.path.join(BASE_DIR, "mascaras")         # Predição 1: SAM com curadoria (Consenso)
DIR_HIBRIDO = os.path.join(BASE_DIR, "predicoes_sam_refinado") # Predição 2: U-Net guiando o SAM

# Pasta de saída para o CSV
DIR_SAIDA = os.path.join(BASE_DIR, "metricas")
CSV_SAIDA = os.path.join(DIR_SAIDA, "comparativo_iou.csv")

# ============================================================
# 2. LÓGICA MATEMÁTICA DO IoU
# ============================================================

def calcular_iou(mascara_predita, mascara_real):
    """
    Calcula o Intersection over Union entre duas matrizes binárias.
    """
    # Binariza estritamente (0 ou 1) para garantir as operações lógicas
    pred_bin = (mascara_predita > 127).astype(bool)
    real_bin = (mascara_real > 127).astype(bool)

    # Interseção: Onde as duas concordam que é desgaste (AND)
    intersecao = np.logical_and(pred_bin, real_bin).sum()
    
    # União: Onde qualquer uma das duas acha que é desgaste (OR)
    uniao = np.logical_or(pred_bin, real_bin).sum()

    # Evita divisão por zero (se ambas as máscaras forem totalmente pretas)
    if uniao == 0:
        # Se as duas acham que não tem desgaste e realmente não tem, acertaram 100%
        return 1.0 if np.sum(real_bin) == 0 else 0.0

    return intersecao / uniao

# ============================================================
# 3. ROTINA DE COMPARAÇÃO
# ============================================================

def buscar_mascara(pasta_base, subpasta, nome_base, prefixos_permitidos):
    """
    Tenta encontrar a máscara correta, lidando com as diferentes nomenclaturas
    que os seus scripts antigos podem ter gerado.
    """
    pasta_busca = os.path.join(pasta_base, subpasta) if subpasta else pasta_base
    
    for prefixo in prefixos_permitidos:
        caminho = os.path.join(pasta_busca, f"{nome_base}{prefixo}")
        if os.path.exists(caminho):
            return cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    return None

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    
    print("Iniciando cálculo de IoU contra o Ground Truth (Anotação Manual)...\n")

    # Prepara o arquivo Excel/CSV
    with open(CSV_SAIDA, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Dataset", "Imagem", "IoU_SAM_Isolado", "IoU_Pipeline_Hibrido"])

    total_imagens_avaliadas = 0

    for pasta in PASTAS_DADOS:
        dir_originais = os.path.join(BASE_DIR, pasta)
        
        if not os.path.exists(dir_originais):
            continue
            
        imagens = [f for f in os.listdir(dir_originais) if f.lower().endswith(('.jpg', '.png'))]
        
        for img_nome in imagens:
            nome_base = os.path.splitext(img_nome)[0]
            
            # 1. Tenta carregar o Ground Truth (Manual)
            mask_manual = buscar_mascara(DIR_MANUAL, pasta, nome_base, ["_mask.png"])
            
            # Se não houver anotação manual para esta imagem, não há como calcular a métrica. Pula.
            if mask_manual is None:
                continue
                
            total_imagens_avaliadas += 1

            # 2. Tenta carregar a predição do SAM (Isolado/Curadoria)
            # Ele procura primeiro pelo consenso, se não achar, procura a padrão
            mask_sam = buscar_mascara(DIR_SAM_ISOLADO, pasta, nome_base, ["_mask_consenso.png", "_mask.png"])
            
            # 3. Tenta carregar a predição do Pipeline Híbrido (U-Net + SAM)
            # Nota: O seu unet_sam.py salva tudo na pasta raiz 'predicoes_sam_refinado', sem dividir em train/test
            mask_hibrido = buscar_mascara(DIR_HIBRIDO, "", nome_base, ["_sam_refinado_mask.png"])

            # Realiza os cálculos
            iou_sam = calcular_iou(mask_sam, mask_manual) if mask_sam is not None else 0.0
            iou_hibrido = calcular_iou(mask_hibrido, mask_manual) if mask_hibrido is not None else 0.0

            # Grava na linha do CSV
            with open(CSV_SAIDA, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                # Substituimos ponto por vírgula no número para o Excel abrir direito no formato brasileiro
                writer.writerow([
                    pasta, 
                    img_nome, 
                    str(round(iou_sam, 4)).replace('.', ','), 
                    str(round(iou_hibrido, 4)).replace('.', ',')
                ])

    print(f"Processo finalizado!")
    print(f"Total de imagens com Ground Truth validadas: {total_imagens_avaliadas}")
    print(f"Arquivo gerado: {CSV_SAIDA}")

if __name__ == "__main__":
    main()