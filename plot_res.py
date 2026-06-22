import os
import cv2
import numpy as np

# ============================================================
# 1. CONFIGURAÇÕES DOS DIRETÓRIOS
# ============================================================

# Define o caminho base do projeto
BASE_DIR = r"C:\Projetos\SAM_V2"

PASTAS_DADOS = ["train", "valid", "test"]

DIR_MANUAL = os.path.join(BASE_DIR, "Anotacao Manual")
DIR_SAM_ISOLADO = os.path.join(BASE_DIR, "mascaras")
DIR_HIBRIDO = os.path.join(BASE_DIR, "predicoes_sam_refinado")

DIR_SAIDA = os.path.join(BASE_DIR, "metricas", "plots_comparativos")

# ============================================================
# 2. FUNÇÕES AUXILIARES DE DESENHO
# ============================================================

def buscar_mascara(pasta_base, subpasta, nome_base, prefixos_permitidos):
    """Busca a máscara correspondente tratando variações de sufixo."""
    pasta_busca = os.path.join(pasta_base, subpasta) if subpasta else pasta_base
    for prefixo in prefixos_permitidos:
        caminho = os.path.join(pasta_busca, f"{nome_base}{prefixo}")
        if os.path.exists(caminho):
            return cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    return None

def desenhar_contorno(imagem, mascara, cor_bgr, espessura=2):
    """Encontra as bordas da máscara e desenha o contorno na imagem."""
    if mascara is None:
        return imagem
        
    # Garante que a máscara seja estritamente binária
    _, binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenha o contorno sobre a imagem
    cv2.drawContours(imagem, contornos, -1, cor_bgr, espessura)
    return imagem

def adicionar_legenda(imagem, texto1, cor1, texto2, cor2):
    """Cria uma barra de legenda no topo da imagem para o TCC."""
    altura_barra = 40
    # Desenha fundo preto semi-transparente no topo
    overlay = imagem.copy()
    cv2.rectangle(overlay, (0, 0), (imagem.shape[1], altura_barra), (0, 0, 0), -1)
    imagem = cv2.addWeighted(overlay, 0.6, imagem, 0.4, 0)

    # Adiciona os textos da legenda
    cv2.putText(imagem, texto1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor1, 2)
    cv2.putText(imagem, texto2, (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor2, 2)
    return imagem

# ============================================================
# 3. ROTINA PRINCIPAL
# ============================================================

def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    print("Iniciando a geração dos plots comparativos para o TCC...\n")

    imagens_geradas = 0

    for pasta in PASTAS_DADOS:
        dir_originais = os.path.join(BASE_DIR, pasta)
        if not os.path.exists(dir_originais): continue
            
        imagens = [f for f in os.listdir(dir_originais) if f.lower().endswith(('.jpg', '.png'))]
        
        for img_nome in imagens:
            nome_base = os.path.splitext(img_nome)[0]
            caminho_original = os.path.join(dir_originais, img_nome)
            
            # Carrega a imagem RGB original
            img_original = cv2.imread(caminho_original)
            if img_original is None: continue

            # Carrega as 3 máscaras
            mask_manual = buscar_mascara(DIR_MANUAL, pasta, nome_base, ["_mask.png"])
            mask_sam = buscar_mascara(DIR_SAM_ISOLADO, pasta, nome_base, ["_mask_consenso.png", "_mask.png"])
            mask_hibrido = buscar_mascara(DIR_HIBRIDO, "", nome_base, ["_sam_refinado_mask.png"])

            # Só gera o plot se houver o Ground Truth (Anotação Manual) para comparar
            if mask_manual is not None:
                # Cores no formato BGR (OpenCV)
                COR_GT = (0, 255, 0)      # Verde (Manual)
                COR_SAM = (0, 255, 255)   # Amarelo (SAM)
                COR_HIBRIDO = (0, 0, 255) # Vermelho (U-Net + SAM)

                # ======================================================
                # PLOT 1: Manual vs SAM Isolado
                # ======================================================
                if mask_sam is not None:
                    plot_sam = img_original.copy()
                    plot_sam = desenhar_contorno(plot_sam, mask_sam, COR_SAM, espessura=2)
                    plot_sam = desenhar_contorno(plot_sam, mask_manual, COR_GT, espessura=2)
                    plot_sam = adicionar_legenda(plot_sam, "GT (Manual)", COR_GT, "SAM Isolado", COR_SAM)
                    
                    cv2.imwrite(os.path.join(DIR_SAIDA, f"{nome_base}_plot_SAM.jpg"), plot_sam)

                # ======================================================
                # PLOT 2: Manual vs Pipeline Híbrido
                # ======================================================
                if mask_hibrido is not None:
                    plot_hib = img_original.copy()
                    plot_hib = desenhar_contorno(plot_hib, mask_hibrido, COR_HIBRIDO, espessura=2)
                    plot_hib = desenhar_contorno(plot_hib, mask_manual, COR_GT, espessura=2)
                    plot_hib = adicionar_legenda(plot_hib, "GT (Manual)", COR_GT, "Pipeline Hibrido", COR_HIBRIDO)
                    
                    cv2.imwrite(os.path.join(DIR_SAIDA, f"{nome_base}_plot_HIBRIDO.jpg"), plot_hib)
                
                imagens_geradas += 1

    print(f"Processo finalizado!")
    print(f"Foram gerados plots comparativos para {imagens_geradas} ferramentas.")
    print(f"Verifique a pasta: {DIR_SAIDA}")

if __name__ == "__main__":
    main()