import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor
import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASTA_TESTE = os.path.join(BASE_DIR, "test")
PASTA_SAIDA = os.path.join(BASE_DIR, "predicoes_sam_refinado")

CAMINHO_MODELO_UNET = os.path.join(BASE_DIR, "modelos", "unet_desgaste.pth")
CAMINHO_MODELO_SAM = os.path.join(BASE_DIR, "modelos", "sam_vit_h_4b8939.pth")

IMG_SIZE = 256
THRESHOLD_UNET = 0.75
MARGEM_BOX = 5

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")

os.makedirs(PASTA_SAIDA, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Usando dispositivo:", device)

# ============================================================
# FUNÇÕES
# ============================================================

def carregar_unet():
    print("Carregando U-Net (SMP ResNet34)...")
    # Instancia a mesma arquitetura usada no treino
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # None porque vamos carregar os SEUS pesos logo abaixo
        in_channels=3,
        classes=1
    ).to(device)
    
    # Carrega os pesos salvos do seu TCC
    model.load_state_dict(torch.load(CAMINHO_MODELO_UNET, map_location=device))
    model.eval()
    print("U-Net carregada com sucesso!")
    return model


def carregar_sam():
    print("Carregando SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=CAMINHO_MODELO_SAM)
    sam.to(device)
    predictor = SamPredictor(sam)
    print("SAM carregado.")
    return predictor


def prever_mascara_unet(model, imagem_bgr):
    altura_original, largura_original = imagem_bgr.shape[:2]

    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_redim = cv2.resize(imagem_rgb, (IMG_SIZE, IMG_SIZE))

    imagem_norm = imagem_redim.astype(np.float32) / 255.0
    imagem_tensor = np.transpose(imagem_norm, (2, 0, 1))
    imagem_tensor = torch.tensor(imagem_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(imagem_tensor)
        prob = torch.sigmoid(logits)
        mascara = (prob > THRESHOLD_UNET).float()

    mascara = mascara.squeeze().cpu().numpy()
    mascara = (mascara * 255).astype(np.uint8)

    mascara = cv2.resize(
        mascara,
        (largura_original, altura_original),
        interpolation=cv2.INTER_NEAREST
    )

    return mascara


def gerar_bounding_box(mascara, largura, altura):
    contours, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    maior_contorno = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(maior_contorno)

    if area < 20:
        return None

    x, y, w, h = cv2.boundingRect(maior_contorno)

    x1 = max(0, x - MARGEM_BOX)
    y1 = max(0, y - MARGEM_BOX)
    x2 = min(largura - 1, x + w + MARGEM_BOX)
    y2 = min(altura - 1, y + h + MARGEM_BOX)

    return np.array([x1, y1, x2, y2])


def refinar_com_sam(predictor, imagem_bgr, input_box):
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    predictor.set_image(imagem_rgb)

    # O SAM retorna 3 máscaras quando multimask_output=True
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True
    )

    # --- IMPLEMENTAÇÃO DO MAJORITY VOTE (VOTO DA MAIORIA) ---
    
    # 1. Garante que as máscaras estão em formato binário (0 ou 1)
    mask1 = masks[0].astype(np.float32)
    mask2 = masks[1].astype(np.float32)
    mask3 = masks[2].astype(np.float32)
    
    # 2. Soma as 3 matrizes de pixels
    soma_mascaras = mask1 + mask2 + mask3
    
    # 3. Consenso: Onde a soma for >= 2, vira 1 (Desgaste). O resto vira 0 (Fundo).
    consenso = np.where(soma_mascaras >= 2, 1.0, 0.0)
    
    # 4. Converte de volta para formato de imagem OpenCV (0 a 255)
    mascara_sam = (consenso * 255).astype(np.uint8)

    # Calcula a média dos scores como referência (já que não há um score único para o consenso)
    score_medio = float(np.mean(scores))

    return mascara_sam, score_medio


def salvar_resultados(nome_base, imagem_bgr, mascara_unet, mascara_sam, input_box, score_sam):
    caminho_unet = os.path.join(PASTA_SAIDA, f"{nome_base}_unet_mask.png")
    caminho_sam = os.path.join(PASTA_SAIDA, f"{nome_base}_sam_refinado_mask.png")
    caminho_overlay = os.path.join(PASTA_SAIDA, f"{nome_base}_overlay_refinado.jpg")
    caminho_box = os.path.join(PASTA_SAIDA, f"{nome_base}_box.jpg")

    pasta_overlay_borda = os.path.join(PASTA_SAIDA, "overlays_com_borda")
    os.makedirs(pasta_overlay_borda, exist_ok=True)

    caminho_overlay_borda = os.path.join(
        pasta_overlay_borda,
        f"{nome_base}_overlay_borda.jpg"
    )

    cv2.imwrite(caminho_unet, mascara_unet)
    cv2.imwrite(caminho_sam, mascara_sam)

    # Overlay vermelho normal
    overlay = imagem_bgr.copy()
    overlay[mascara_sam > 0] = [0, 0, 255]

    combinado = cv2.addWeighted(imagem_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite(caminho_overlay, combinado)

    # Overlay com borda
    overlay_borda = combinado.copy()

    contours, _ = cv2.findContours(
        mascara_sam,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay_borda,
        contours,
        -1,
        (0, 255, 255),  # branco
        1
    )

    cv2.imwrite(caminho_overlay_borda, overlay_borda)

    # Imagem com bounding box
    imagem_box = imagem_bgr.copy()
    x1, y1, x2, y2 = input_box.astype(int)
    cv2.rectangle(imagem_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(caminho_box, imagem_box)

    print(f"Salvo: {nome_base}")
    print(f"Score SAM: {score_sam:.4f}")

    # ============================================================
    # IMAGEM COMPARATIVA LADO A LADO
    # ============================================================

    pasta_comparativo = os.path.join(PASTA_SAIDA, "comparativos")
    os.makedirs(pasta_comparativo, exist_ok=True)

    caminho_comparativo = os.path.join(
        pasta_comparativo,
        f"{nome_base}_comparativo.jpg"
    )

    # Garantir mesmo tamanho (por segurança)
    h, w = imagem_bgr.shape[:2]
    overlay_resized = cv2.resize(overlay_borda, (w, h))

    # Concatenar horizontalmente
    comparativo = np.hstack((imagem_bgr, overlay_resized))

    cv2.imwrite(caminho_comparativo, comparativo)

# ============================================================
# MAIN
# ============================================================

def main():
    unet = carregar_unet()
    predictor = carregar_sam()

    imagens = [
        arq for arq in os.listdir(PASTA_TESTE)
        if arq.lower().endswith(EXTENSOES)
    ]

    imagens.sort()

    print(f"\n{len(imagens)} imagem(ns) encontradas em test.")

    for nome_imagem in imagens:
        caminho_imagem = os.path.join(PASTA_TESTE, nome_imagem)

        imagem_bgr = cv2.imread(caminho_imagem)

        if imagem_bgr is None:
            print("Erro ao abrir:", nome_imagem)
            continue

        altura, largura = imagem_bgr.shape[:2]
        nome_base = os.path.splitext(nome_imagem)[0]

        print("\nProcessando:", nome_imagem)

        mascara_unet = prever_mascara_unet(unet, imagem_bgr)

        input_box = gerar_bounding_box(mascara_unet, largura, altura)

        if input_box is None:
            print("Nenhuma região detectada pela U-Net. Pulando imagem.")
            continue

        mascara_sam, score_sam = refinar_com_sam(
            predictor,
            imagem_bgr,
            input_box
        )

        salvar_resultados(
            nome_base,
            imagem_bgr,
            mascara_unet,
            mascara_sam,
            input_box,
            score_sam
        )

    print("\nProcesso finalizado.")
    print("Resultados salvos em:", PASTA_SAIDA)


if __name__ == "__main__":
    main()