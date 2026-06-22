import os
import cv2
import csv
import sys
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# 1. CONFIGURAÇÕES PRINCIPAIS DO PROJETO
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    PASTAS_DATASET = [sys.argv[1]]
else:
    PASTAS_DATASET = ["train", "valid", "test"]

CAMINHO_MODELO = os.path.join(BASE_DIR, "modelos", "sam_vit_h_4b8939.pth")

PASTA_MASCARAS_BASE = os.path.join(BASE_DIR, "mascaras")
PASTA_OVERLAYS_BASE = os.path.join(BASE_DIR, "overlays")
PASTA_RESULTADOS = os.path.join(BASE_DIR, "resultados")

CAMINHO_CSV = os.path.join(PASTA_RESULTADOS, "scores_sam.csv")

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")

os.makedirs(PASTA_MASCARAS_BASE, exist_ok=True)
os.makedirs(PASTA_OVERLAYS_BASE, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# ============================================================
# 2. CARREGAMENTO DO MODELO SAM
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Carregando SAM...")
sam = sam_model_registry["vit_h"](checkpoint=CAMINHO_MODELO)
sam.to(device)

predictor = SamPredictor(sam)

print("Modelo carregado em:", device)

# ============================================================
# 3. FUNÇÕES AUXILIARES DE IMAGEM
# ============================================================

def redimensionar_para_tela(imagem, max_largura=1200, max_altura=800):
    altura, largura = imagem.shape[:2]
    escala = min(max_largura / largura, max_altura / altura)
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    imagem_display = cv2.resize(imagem, (nova_largura, nova_altura))
    return imagem_display, escala

def criar_overlay(imagem_original, mascara, texto, alpha=0.35, espessura_contorno=1):
    imagem_overlay = imagem_original.copy()
    mascara_bin = mascara > 127

    camada_vermelha = np.zeros_like(imagem_original)
    camada_vermelha[mascara_bin] = (0, 0, 255)

    imagem_overlay = cv2.addWeighted(imagem_overlay, 1.0, camada_vermelha, alpha, 0)

    contornos, _ = cv2.findContours(mascara_bin.astype("uint8") * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagem_overlay, contornos, -1, (0, 255, 255), espessura_contorno)

    cv2.putText(imagem_overlay, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return imagem_overlay

# ============================================================
# 4. FUNÇÕES DE SALVAMENTO
# ============================================================

def inicializar_csv():
    if not os.path.exists(CAMINHO_CSV):
        with open(CAMINHO_CSV, mode="w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo, delimiter=";")
            escritor.writerow(["pasta", "imagem", "nome_base", "candidata", "score", "arquivo_mascara", "arquivo_overlay", "tipo_registro"])

def registrar_csv(pasta, nome_imagem, nome_base, candidata, score, arquivo_mascara, arquivo_overlay, tipo_registro):
    with open(CAMINHO_CSV, mode="a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow([pasta, nome_imagem, nome_base, candidata, f"{score:.4f}" if score is not None else "", arquivo_mascara, arquivo_overlay, tipo_registro])

def salvar_mascaras_e_overlays(pasta, nome_imagem, nome_base, imagem_original, masks, scores, pasta_mascaras, pasta_overlays):
    indices_ordenados = np.argsort(scores)[::-1]
    arquivos_salvos = []
    
    # Lista para guardar as máscaras binárias e calcular o consenso depois
    mascaras_binarias = []

    # 1. Salva as 3 candidatas
    for rank, idx in enumerate(indices_ordenados, start=1):
        score = float(scores[idx])
        mascara = (masks[idx] * 255).astype("uint8")
        mascaras_binarias.append(mascara) # Guarda para o consenso

        nome_mascara = f"{nome_base}_mask_{rank:02d}_score_{score:.4f}.png"
        nome_overlay = f"{nome_base}_overlay_{rank:02d}_score_{score:.4f}.png"

        caminho_mascara = os.path.join(pasta_mascaras, nome_mascara)
        caminho_overlay = os.path.join(pasta_overlays, nome_overlay)

        cv2.imwrite(caminho_mascara, mascara)

        texto_overlay = f"CANDIDATA {rank} | SCORE: {score:.4f}"
        overlay = criar_overlay(imagem_original, mascara, texto_overlay)
        cv2.imwrite(caminho_overlay, overlay)

        registrar_csv(pasta, nome_imagem, nome_base, rank, score, nome_mascara, nome_overlay, "sam_multimask")
        arquivos_salvos.append((rank, score, caminho_mascara, caminho_overlay))

   # 2. GERAÇÃO DA MÁSCARA DE CONSENSO (VOTO DA MAIORIA: 2 de 3)
    if len(mascaras_binarias) >= 3:
        # Converte de 0-255 para 0-1 e soma as matrizes
        soma = (mascaras_binarias[0] / 255.0) + \
               (mascaras_binarias[1] / 255.0) + \
               (mascaras_binarias[2] / 255.0)
        
        # Voto da Maioria: Onde a soma for >= 2 (ou seja, 2 ou 3 candidatas concordam), vira 255
        consenso_final = np.where(soma >= 2, 255, 0).astype("uint8")
        
        nome_mascara_consenso = f"{nome_base}_mask_consenso.png"
        nome_overlay_consenso = f"{nome_base}_overlay_consenso.png"
        
        caminho_mascara_consenso = os.path.join(pasta_mascaras, nome_mascara_consenso)
        caminho_overlay_consenso = os.path.join(pasta_overlays, nome_overlay_consenso)
        
        # Salva a máscara de consenso
        cv2.imwrite(caminho_mascara_consenso, consenso_final)
        
        # Cria overlay do consenso
        overlay_cons = criar_overlay(imagem_original, consenso_final, "CONSENSO (VOTO MAIORIA)")
        cv2.imwrite(caminho_overlay_consenso, overlay_cons)
        
        # Registra no CSV como rank 0 (Consenso)
        registrar_csv(pasta, nome_imagem, nome_base, 0, None, nome_mascara_consenso, nome_overlay_consenso, "consenso_maioria")
        arquivos_salvos.append(("Consenso", 0.0, caminho_mascara_consenso, caminho_overlay_consenso))

    # 3. Compatibilidade Legada (salva a melhor como _mask.png)
    melhor_idx = indices_ordenados[0]
    melhor_mascara = (masks[melhor_idx] * 255).astype("uint8")
    caminho_legado = os.path.join(pasta_mascaras, f"{nome_base}_mask.png")
    cv2.imwrite(caminho_legado, melhor_mascara)

    return arquivos_salvos

def salvar_mascaras_vazias(pasta, nome_imagem, nome_base, imagem_original, pasta_mascaras, pasta_overlays):
    altura, largura = imagem_original.shape[:2]
    mascara_vazia = np.zeros((altura, largura), dtype="uint8")

    caminho_legado = os.path.join(pasta_mascaras, f"{nome_base}_mask.png")
    cv2.imwrite(caminho_legado, mascara_vazia)

    for rank in range(1, 4):
        score = 0.0
        nome_mascara = f"{nome_base}_mask_{rank:02d}_score_{score:.4f}.png"
        nome_overlay = f"{nome_base}_overlay_{rank:02d}_score_{score:.4f}.png"

        caminho_mascara = os.path.join(pasta_mascaras, nome_mascara)
        caminho_overlay = os.path.join(pasta_overlays, nome_overlay)

        cv2.imwrite(caminho_mascara, mascara_vazia)
        overlay = criar_overlay(imagem_original, mascara_vazia, f"CANDIDATA {rank} | SCORE: {score:.4f}")
        cv2.imwrite(caminho_overlay, overlay)

        registrar_csv(pasta, nome_imagem, nome_base, rank, score, nome_mascara, nome_overlay, "mascara_vazia")

# ============================================================
# 5. PROCESSAMENTO INTERATIVO DE CADA IMAGEM
# ============================================================

def processar_imagem(nome_pasta, nome_imagem, pasta_imagens, pasta_mascaras, pasta_overlays):
    caminho_imagem = os.path.join(pasta_imagens, nome_imagem)
    imagem_original = cv2.imread(caminho_imagem)

    if imagem_original is None:
        print("Erro ao abrir:", nome_imagem)
        return

    imagem_rgb = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
    predictor.set_image(imagem_rgb)

    imagem_display, escala = redimensionar_para_tela(imagem_original)

    caixa_inicio = None
    caixa_fim = None
    desenhando_caixa = False
    salvar_vazia = False
    pular_imagem = False

    def desenhar_interface():
        tela_atual = imagem_display.copy()
        if caixa_inicio is not None and caixa_fim is not None:
            cv2.rectangle(tela_atual, caixa_inicio, caixa_fim, (255, 0, 0), 2)
        cv2.imshow("SAM - Bounding Box", tela_atual)

    def mouse_callback(event, x, y, flags, param):
        nonlocal caixa_inicio, caixa_fim, desenhando_caixa
        if event == cv2.EVENT_LBUTTONDOWN:
            caixa_inicio = (x, y)
            caixa_fim = (x, y)
            desenhando_caixa = True
            desenhar_interface()
        elif event == cv2.EVENT_MOUSEMOVE and desenhando_caixa:
            caixa_fim = (x, y)
            desenhar_interface()
        elif event == cv2.EVENT_LBUTTONUP:
            caixa_fim = (x, y)
            desenhando_caixa = False
            desenhar_interface()

    cv2.imshow("SAM - Bounding Box", imagem_display)
    cv2.setMouseCallback("SAM - Bounding Box", mouse_callback)

    print(f"\nImagem: {nome_imagem} | S=Segmentar | B=Vazio | N=Pular | R=Reset | ESC=Sair")

    while True:
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("s"): break
        elif tecla == ord("b"): salvar_vazia = True; break
        elif tecla == ord("n"): pular_imagem = True; break
        elif tecla == ord("r"):
            caixa_inicio = None; caixa_fim = None; desenhando_caixa = False
            desenhar_interface()
        elif tecla == 27:
            cv2.destroyAllWindows(); exit()

    cv2.destroyAllWindows()
    nome_base = os.path.splitext(nome_imagem)[0]

    if pular_imagem: return
    if salvar_vazia:
        salvar_mascaras_vazias(nome_pasta, nome_imagem, nome_base, imagem_original, pasta_mascaras, pasta_overlays)
        return

    if caixa_inicio is None or caixa_fim is None: return

    x1 = int(min(caixa_inicio[0], caixa_fim[0]) / escala)
    y1 = int(min(caixa_inicio[1], caixa_fim[1]) / escala)
    x2 = int(max(caixa_inicio[0], caixa_fim[0]) / escala)
    y2 = int(max(caixa_inicio[1], caixa_fim[1]) / escala)

    input_box = np.array([x1, y1, x2, y2])

    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=True)

    salvar_mascaras_e_overlays(nome_pasta, nome_imagem, nome_base, imagem_original, masks, scores, pasta_mascaras, pasta_overlays)

# ============================================================
# 6. LOOP PRINCIPAL DO SCRIPT
# ============================================================

inicializar_csv()

for nome_pasta in PASTAS_DATASET:
    pasta_imagens = os.path.join(BASE_DIR, nome_pasta)
    pasta_mascaras = os.path.join(PASTA_MASCARAS_BASE, nome_pasta)
    pasta_overlays = os.path.join(PASTA_OVERLAYS_BASE, nome_pasta)

    os.makedirs(pasta_mascaras, exist_ok=True)
    os.makedirs(pasta_overlays, exist_ok=True)

    if not os.path.exists(pasta_imagens): continue

    imagens = [arq for arq in os.listdir(pasta_imagens) if arq.lower().endswith(EXTENSOES)]
    imagens.sort()

    #imagens = imagens[13:]

    print(f"\n=== Processando {nome_pasta} ===")
    
    for nome_imagem in imagens:
        processar_imagem(nome_pasta, nome_imagem, pasta_imagens, pasta_mascaras, pasta_overlays)

print("\nProcesso finalizado.")