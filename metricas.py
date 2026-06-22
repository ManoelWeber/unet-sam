import os
import cv2
import numpy as np
import csv

# ============================================================
# CONFIGURAÇÕES DE CAMINHOS
# ============================================================

BASE_DIR = r"C:\Projetos\SAM_V2"

PASTA_MANUAL_BASE = os.path.join(BASE_DIR, "Anotacao Manual")
PASTA_SAM_BASE = os.path.join(BASE_DIR, "mascaras")
PASTA_HIBRIDO_BASE = os.path.join(BASE_DIR, "predicoes_sam_refinado")

PASTA_SAIDA_METRICAS = os.path.join(BASE_DIR, "metricas")
PASTA_PLOTS = os.path.join(PASTA_SAIDA_METRICAS, "plots_comparativos")

os.makedirs(PASTA_SAIDA_METRICAS, exist_ok=True)
os.makedirs(PASTA_PLOTS, exist_ok=True)

CSV_SAIDA = os.path.join(PASTA_SAIDA_METRICAS, "metricas_completas_com_plots.csv")

# Fator de aumento da imagem final.
# 2 = dobra a resolução do output.
# 3 = triplica a resolução do output.
FATOR_UPSCALE_PLOT = 4

# Qualidade do JPG de saída
QUALIDADE_JPG = 95


# ============================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# ============================================================

def binarizar_mascara(mascara):
    """
    Converte a máscara para binária:
    - Pixels acima de 127 viram 255
    - Pixels abaixo ou iguais a 127 viram 0
    """
    return (mascara > 127).astype(np.uint8) * 255


def redimensionar_para_referencia(imagem, referencia, is_mask=False):
    """
    Redimensiona uma imagem/máscara para o mesmo tamanho da imagem de referência.
    Para máscaras, usa interpolação NEAREST para não criar tons intermediários.
    """
    if imagem is None or referencia is None:
        return imagem

    h_ref, w_ref = referencia.shape[:2]
    h_img, w_img = imagem.shape[:2]

    if (h_img, w_img) == (h_ref, w_ref):
        return imagem

    interpolacao = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(imagem, (w_ref, h_ref), interpolation=interpolacao)


def calcular_area_e_profundidade(mascara):
    """
    Calcula:
    - Área da máscara em pixels
    - Profundidade/VB em pixels, considerando a altura do maior contorno
    - Bounding box do maior contorno
    """
    mascara_bin = binarizar_mascara(mascara)
    area_pixels = int(np.sum(mascara_bin == 255))

    contornos, _ = cv2.findContours(
        mascara_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    profundidade = 0
    bbox = None

    if contornos:
        maior_contorno = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maior_contorno)

        profundidade = int(h)
        bbox = (int(x), int(y), int(w), int(h))

    return area_pixels, profundidade, bbox


def calcular_metricas_segmentacao(mascara_pred, mascara_gt):
    """
    Calcula métricas clássicas de segmentação binária:
    - IoU
    - Precision
    - Recall
    - F1-Score

    Fórmulas:
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1-Score  = 2 * Precision * Recall / (Precision + Recall)
    IoU       = TP / (TP + FP + FN)
    """
    pred_bin = mascara_pred > 127
    gt_bin = mascara_gt > 127

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()
    tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score)
    }


def extrair_borda_mascara(mascara):
    """
    Extrai apenas a borda externa da máscara binária.

    A Distância de Hausdorff é calculada sobre essas bordas, pois ela avalia
    o maior afastamento entre os contornos da máscara predita e da máscara manual.
    """
    mascara_bin = binarizar_mascara(mascara)
    borda = np.zeros_like(mascara_bin, dtype=np.uint8)

    contornos, _ = cv2.findContours(
        mascara_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contornos:
        cv2.drawContours(borda, contornos, -1, 255, thickness=1)

    return borda


def calcular_distancia_hausdorff(mascara_pred, mascara_gt):
    """
    Calcula a Distância de Hausdorff simétrica entre as bordas das máscaras.

    Interpretação:
    - Valor menor indica maior proximidade entre os contornos.
    - Valor maior indica maior afastamento máximo entre as bordas.
    - Unidade: pixels.

    Observação:
    - Se as duas máscaras estiverem vazias, retorna 0.
    - Se apenas uma máscara estiver vazia, retorna NaN, pois a distância não é definida.
    """
    borda_pred = extrair_borda_mascara(mascara_pred)
    borda_gt = extrair_borda_mascara(mascara_gt)

    pred_pts = np.column_stack(np.where(borda_pred > 0))
    gt_pts = np.column_stack(np.where(borda_gt > 0))

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    # cv2.distanceTransform calcula, para cada pixel não-zero, a distância até o pixel zero mais próximo.
    # Por isso, a borda de referência é marcada como 0 e o restante como 255.
    dist_para_gt = cv2.distanceTransform(
        255 - borda_gt,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    dist_para_pred = cv2.distanceTransform(
        255 - borda_pred,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    # Os pontos estão no formato (linha, coluna), equivalente a (y, x).
    hd_pred_gt = np.max(dist_para_gt[pred_pts[:, 0], pred_pts[:, 1]])
    hd_gt_pred = np.max(dist_para_pred[gt_pts[:, 0], gt_pts[:, 1]])

    return float(max(hd_pred_gt, hd_gt_pred))


def formatar_hausdorff(valor):
    """Formata a Distância de Hausdorff para exibição no CSV e no cabeçalho."""
    if valor is None or not np.isfinite(valor):
        return "N/A"

    return f"{valor:.2f}"


def calcular_diferencas(valor_manual, valor_predito):
    """
    Calcula diferença absoluta e percentual entre valor predito e manual.

    Diferença absoluta:
    Predito - Manual

    Diferença percentual:
    ((Predito - Manual) / Manual) * 100
    """
    diferenca_abs = valor_predito - valor_manual

    if valor_manual != 0:
        diferenca_perc = (diferenca_abs / valor_manual) * 100
    else:
        diferenca_perc = 0.0

    return diferenca_abs, diferenca_perc


def carregar_imagem(pasta, nome_base, sufixos, is_color=False):
    """
    Tenta carregar uma imagem usando uma lista de possíveis sufixos.
    """
    flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE

    for sufixo in sufixos:
        caminho = os.path.join(pasta, f"{nome_base}{sufixo}")

        if os.path.exists(caminho):
            return cv2.imread(caminho, flag)

    return None


# ============================================================
# FUNÇÕES DE PLOTAGEM
# ============================================================

def aplicar_upscale(imagem, fator=2, is_mask=False):
    """
    Aumenta a resolução da imagem.
    Para máscaras, usa INTER_NEAREST.
    Para imagem colorida, usa INTER_CUBIC.
    """
    if fator <= 1:
        return imagem

    h, w = imagem.shape[:2]
    novo_w = int(w * fator)
    novo_h = int(h * fator)

    interpolacao = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC

    return cv2.resize(imagem, (novo_w, novo_h), interpolation=interpolacao)


def gerar_plot_comparativo(
    img_original,
    mask_gt,
    mask_pred,
    area_gt,
    area_pred,
    prof_gt,
    prof_pred,
    metricas,
    diff_area_abs,
    diff_area_perc,
    diff_vb_abs,
    diff_vb_perc,
    hausdorff,
    cor_pred,
    caminho_salvar,
    nome_modelo,
    fator_upscale=2
):
    """
    Gera imagem comparativa entre máscara manual e máscara predita.

    Melhorias aplicadas:
    - Upscaling da imagem, máscaras e contornos
    - Fonte maior e mais legível
    - Cabeçalho com métricas principais
    - Exibição de diferença de área, VB e Distância de Hausdorff
    """

    # ------------------------------------------------------------
    # 1. Upscaling da imagem e das máscaras
    # ------------------------------------------------------------

    img_plot = aplicar_upscale(img_original.copy(), fator_upscale, is_mask=False)
    mask_gt_up = aplicar_upscale(mask_gt, fator_upscale, is_mask=True)
    mask_pred_up = aplicar_upscale(mask_pred, fator_upscale, is_mask=True)

    # Recalcula os bboxes após o upscale
    _, prof_gt_up, bbox_gt_up = calcular_area_e_profundidade(mask_gt_up)
    _, prof_pred_up, bbox_pred_up = calcular_area_e_profundidade(mask_pred_up)

    # ------------------------------------------------------------
    # 2. Configurações visuais
    # ------------------------------------------------------------

    fonte = cv2.FONT_HERSHEY_DUPLEX

    escala_fonte = 0.56 * fator_upscale
    escala_fonte_menor = 0.32 * fator_upscale
    escala_fonte_geometrica = 0.30 * fator_upscale

    espessura_fonte = max(1, int(1 * fator_upscale))
    espessura_linha = max(1, int(1.5 * fator_upscale))
    espessura_contorno = max(1, int(1.2 * fator_upscale))

    cor_gt = (255, 0, 0)        # Azul - Manual
    cor_branco = (255, 255, 255)
    cor_cinza = (210, 210, 210)
    cor_fundo = (0, 0, 0)

    # ------------------------------------------------------------
    # 3. Desenho dos contornos
    # ------------------------------------------------------------

    contornos_gt, _ = cv2.findContours(
        binarizar_mascara(mask_gt_up),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_pred, _ = cv2.findContours(
        binarizar_mascara(mask_pred_up),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(img_plot, contornos_pred, -1, cor_pred, espessura_contorno)
    cv2.drawContours(img_plot, contornos_gt, -1, cor_gt, espessura_contorno)

    # ------------------------------------------------------------
    # 4. Cota do VB Manual
    # ------------------------------------------------------------

    if bbox_gt_up is not None:
        x, y, w, h = bbox_gt_up
        x_esq = max(20, x - int(35 * fator_upscale))

        cv2.line(img_plot, (x_esq, y), (x_esq, y + h), cor_gt, espessura_linha)
        cv2.line(img_plot, (x_esq - 8, y), (x_esq + 8, y), cor_gt, espessura_linha)
        cv2.line(img_plot, (x_esq - 8, y + h), (x_esq + 8, y + h), cor_gt, espessura_linha)

        cv2.putText(
            img_plot,
            f"VB Man: {prof_gt}px",
            (max(10, x_esq - int(120 * fator_upscale)), y + int(h / 2)),
            fonte,
            escala_fonte_menor,
            cor_gt,
            espessura_fonte
        )

    # ------------------------------------------------------------
    # 5. Cota do VB Predito
    # ------------------------------------------------------------

    if bbox_pred_up is not None:
        x, y, w, h = bbox_pred_up
        x_dir = min(img_plot.shape[1] - 20, x + w + int(35 * fator_upscale))

        cv2.line(img_plot, (x_dir, y), (x_dir, y + h), cor_pred, espessura_linha)
        cv2.line(img_plot, (x_dir - 8, y), (x_dir + 8, y), cor_pred, espessura_linha)
        cv2.line(img_plot, (x_dir - 8, y + h), (x_dir + 8, y + h), cor_pred, espessura_linha)

        cv2.putText(
            img_plot,
            f"VB Pred: {prof_pred}px",
            (min(img_plot.shape[1] - int(230 * fator_upscale), x_dir + int(12 * fator_upscale)),
             y + int(h / 2)),
            fonte,
            escala_fonte_menor,
            cor_pred,
            espessura_fonte
        )

    # ------------------------------------------------------------
    # 6. Cabeçalho superior
    # ------------------------------------------------------------

    altura_cabecalho = int(145 * fator_upscale)
    largura = img_plot.shape[1]

    fundo_texto = np.zeros((altura_cabecalho, largura, 3), dtype=np.uint8)
    fundo_texto[:] = cor_fundo

    img_plot = np.vstack((fundo_texto, img_plot))

    y1 = int(30 * fator_upscale)
    y2 = int(62 * fator_upscale)
    y3 = int(96 * fator_upscale)
    y4 = int(124 * fator_upscale)

    x0 = int(12 * fator_upscale)

    # Linha 1: legenda
    cv2.putText(
        img_plot,
        "Legenda:",
        (x0, y1),
        fonte,
        escala_fonte,
        cor_branco,
        espessura_fonte
    )

    cv2.putText(
        img_plot,
        "Manual",
        (int(115 * fator_upscale), y1),
        fonte,
        escala_fonte,
        cor_gt,
        espessura_fonte
    )

    cv2.putText(
        img_plot,
        f"| {nome_modelo}",
        (int(230 * fator_upscale), y1),
        fonte,
        escala_fonte,
        cor_pred,
        espessura_fonte
    )

    # Linha 2: métricas principais, incluindo Distância de Hausdorff
    texto_metricas = (
        f"IoU: {metricas['IoU'] * 100:.2f}% | "
        f"Prec: {metricas['Precision'] * 100:.2f}% | "
        f"Rec: {metricas['Recall'] * 100:.2f}% | "
        f"F1: {metricas['F1-Score'] * 100:.2f}% | "
        f"HD: {formatar_hausdorff(hausdorff)} px"
    )

    cv2.putText(
        img_plot,
        texto_metricas,
        (x0, y2),
        fonte,
        escala_fonte_menor,
        cor_cinza,
        espessura_fonte
    )

    # Linha 3: área e VB
    texto_area_vb = (
        f"Area Man: {area_gt}px | Area {nome_modelo}: {area_pred}px | "
        f"VB Man: {prof_gt}px | VB {nome_modelo}: {prof_pred}px"
    )

    cv2.putText(
        img_plot,
        texto_area_vb,
        (x0, y3),
        fonte,
        escala_fonte_geometrica,
        cor_cinza,
        espessura_fonte
    )

    # Linha 4: diferenças geométricas
    texto_diferencas = (
        f"Dif Area: {diff_area_abs:+d}px | DPAA: {diff_area_perc:+.2f}% | "
        f"Dif VB: {diff_vb_abs:+d}px | DPAVB: {diff_vb_perc:+.2f}%"
    )

    cv2.putText(
        img_plot,
        texto_diferencas,
        (x0, y4),
        fonte,
        escala_fonte_geometrica,
        cor_cinza,
        espessura_fonte
    )

    # ------------------------------------------------------------
    # 7. Salvar imagem
    # ------------------------------------------------------------

    cv2.imwrite(
        caminho_salvar,
        img_plot,
        [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPG]
    )


# ============================================================
# PROCESSAMENTO DE CADA MODELO
# ============================================================

def processar_modelo(
    nome_modelo,
    mask_pred,
    mask_manual,
    img_original,
    area_man,
    prof_man,
    subset,
    nome_base,
    pasta_plots
):
    """
    Processa uma máscara predita em relação à máscara manual.
    Retorna métricas, diferenças e gera o plot comparativo.
    """

    if mask_pred is None:
        return {
            "area": 0,
            "prof": 0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hausdorff": float("nan"),
            "diff_area_abs": 0,
            "diff_area_perc": 0.0,
            "diff_vb_abs": 0,
            "diff_vb_perc": 0.0
        }

    # Garante que a máscara predita tenha o mesmo tamanho da máscara manual
    mask_pred = redimensionar_para_referencia(mask_pred, mask_manual, is_mask=True)

    area_pred, prof_pred, _ = calcular_area_e_profundidade(mask_pred)

    metricas = calcular_metricas_segmentacao(mask_pred, mask_manual)
    hausdorff = calcular_distancia_hausdorff(mask_pred, mask_manual)

    diff_area_abs, diff_area_perc = calcular_diferencas(area_man, area_pred)
    diff_vb_abs, diff_vb_perc = calcular_diferencas(prof_man, prof_pred)

    caminho_plot = os.path.join(
        pasta_plots,
        f"{subset}_{nome_base}_plot_{nome_modelo}.jpg"
    )

    gerar_plot_comparativo(
        img_original=img_original,
        mask_gt=mask_manual,
        mask_pred=mask_pred,
        area_gt=area_man,
        area_pred=area_pred,
        prof_gt=prof_man,
        prof_pred=prof_pred,
        metricas=metricas,
        diff_area_abs=diff_area_abs,
        diff_area_perc=diff_area_perc,
        diff_vb_abs=diff_vb_abs,
        diff_vb_perc=diff_vb_perc,
        hausdorff=hausdorff,
        cor_pred=(0, 255, 255),  # Amarelo em BGR
        caminho_salvar=caminho_plot,
        nome_modelo=nome_modelo,
        fator_upscale=FATOR_UPSCALE_PLOT
    )

    return {
        "area": area_pred,
        "prof": prof_pred,
        "iou": metricas["IoU"] * 100,
        "precision": metricas["Precision"] * 100,
        "recall": metricas["Recall"] * 100,
        "f1": metricas["F1-Score"] * 100,
        "hausdorff": hausdorff,
        "diff_area_abs": diff_area_abs,
        "diff_area_perc": diff_area_perc,
        "diff_vb_abs": diff_vb_abs,
        "diff_vb_perc": diff_vb_perc
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Iniciando cálculo de métricas completas e geração dos plots...")
    print("-" * 70)

    with open(CSV_SAIDA, mode="w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f, delimiter=";")

        escritor.writerow([
            "Dataset",
            "Nome da Imagem",

            "Area Manual (px)",
            "VB Manual (px)",

            "Area SAM (px)",
            "VB SAM (px)",
            "Dif Area SAM (px)",
            "DPAA SAM (%)",
            "Dif VB SAM (px)",
            "DPAVB SAM (%)",
            "Hausdorff SAM (px)",
            "IoU SAM (%)",
            "Precision SAM (%)",
            "Recall SAM (%)",
            "F1-Score SAM (%)",

            "Area SAM+U-Net (px)",
            "VB SAM+U-Net (px)",
            "Dif Area SAM+U-Net (px)",
            "DPAA SAM+U-Net (%)",
            "Dif VB SAM+U-Net (px)",
            "DPAVB SAM+U-Net (%)",
            "Hausdorff SAM+U-Net (px)",
            "IoU SAM+U-Net (%)",
            "Precision SAM+U-Net (%)",
            "Recall SAM+U-Net (%)",
            "F1-Score SAM+U-Net (%)"
        ])

        subsets = ["test", "train"]

        for subset in subsets:
            pasta_manual = os.path.join(PASTA_MANUAL_BASE, subset)
            pasta_sam = os.path.join(PASTA_SAM_BASE, subset)
            pasta_original = os.path.join(BASE_DIR, subset)

            if not os.path.exists(pasta_manual):
                print(f"[AVISO] Pasta manual não encontrada: {pasta_manual}")
                continue

            arquivos_manuais = sorted([
                arq for arq in os.listdir(pasta_manual)
                if arq.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])

            # Mantém sua regra original: limita o train às primeiras 20 imagens
            if subset == "train":
                arquivos_manuais = arquivos_manuais[:20]

            for arquivo in arquivos_manuais:
                nome_base = (
                    arquivo
                    .replace("_mask.png", "")
                    .replace("_mask.jpg", "")
                    .replace("_mask.jpeg", "")
                    .replace(".png", "")
                    .replace(".jpg", "")
                    .replace(".jpeg", "")
                    .replace(".bmp", "")
                )

                print(f"Processando [{subset.upper()}]: {nome_base}")

                # ------------------------------------------------------------
                # Carregar máscara manual e imagem original
                # ------------------------------------------------------------

                mask_manual = carregar_imagem(
                    pasta_manual,
                    nome_base,
                    ["_mask.png", "_mask.jpg", ".png", ".jpg", ".jpeg", ".bmp"],
                    is_color=False
                )

                img_original = carregar_imagem(
                    pasta_original,
                    nome_base,
                    [".jpg", ".jpeg", ".png", ".bmp"],
                    is_color=True
                )

                if mask_manual is None:
                    print(f"[AVISO] Máscara manual não encontrada: {nome_base}")
                    continue

                if img_original is None:
                    print(f"[AVISO] Imagem original não encontrada: {nome_base}")
                    continue

                # Garante que a máscara manual tenha o mesmo tamanho da imagem original
                mask_manual = redimensionar_para_referencia(
                    mask_manual,
                    img_original,
                    is_mask=True
                )

                area_man, prof_man, _ = calcular_area_e_profundidade(mask_manual)

                # ------------------------------------------------------------
                # Processar SAM
                # ------------------------------------------------------------

                mask_sam = carregar_imagem(
                    pasta_sam,
                    nome_base,
                    ["_mask_consenso.png", "_mask.png", ".png", ".jpg"],
                    is_color=False
                )

                resultado_sam = processar_modelo(
                    nome_modelo="SAM",
                    mask_pred=mask_sam,
                    mask_manual=mask_manual,
                    img_original=img_original,
                    area_man=area_man,
                    prof_man=prof_man,
                    subset=subset,
                    nome_base=nome_base,
                    pasta_plots=PASTA_PLOTS
                )

                # ------------------------------------------------------------
                # Processar SAM + U-Net
                # ------------------------------------------------------------
                # Mantive compatibilidade com duas possibilidades:
                # 1) predicoes_sam_refinado/test/nome_arquivo
                # 2) predicoes_sam_refinado/nome_arquivo

                pasta_hibrido_subset = os.path.join(PASTA_HIBRIDO_BASE, subset)

                mask_hibrido = carregar_imagem(
                    pasta_hibrido_subset,
                    nome_base,
                    ["_sam_refinado_mask.png", "_unet_mask.png", ".png", ".jpg"],
                    is_color=False
                )

                if mask_hibrido is None:
                    mask_hibrido = carregar_imagem(
                        PASTA_HIBRIDO_BASE,
                        nome_base,
                        ["_sam_refinado_mask.png", "_unet_mask.png", ".png", ".jpg"],
                        is_color=False
                    )

                resultado_hib = processar_modelo(
                    nome_modelo="SAM+U-Net",
                    mask_pred=mask_hibrido,
                    mask_manual=mask_manual,
                    img_original=img_original,
                    area_man=area_man,
                    prof_man=prof_man,
                    subset=subset,
                    nome_base=nome_base,
                    pasta_plots=PASTA_PLOTS
                )

                # ------------------------------------------------------------
                # Escrever linha no CSV
                # ------------------------------------------------------------

                escritor.writerow([
                    subset.upper(),
                    nome_base,

                    area_man,
                    prof_man,

                    resultado_sam["area"],
                    resultado_sam["prof"],
                    resultado_sam["diff_area_abs"],
                    f"{resultado_sam['diff_area_perc']:.2f}",
                    resultado_sam["diff_vb_abs"],
                    f"{resultado_sam['diff_vb_perc']:.2f}",
                    formatar_hausdorff(resultado_sam["hausdorff"]),
                    f"{resultado_sam['iou']:.2f}",
                    f"{resultado_sam['precision']:.2f}",
                    f"{resultado_sam['recall']:.2f}",
                    f"{resultado_sam['f1']:.2f}",

                    resultado_hib["area"],
                    resultado_hib["prof"],
                    resultado_hib["diff_area_abs"],
                    f"{resultado_hib['diff_area_perc']:.2f}",
                    resultado_hib["diff_vb_abs"],
                    f"{resultado_hib['diff_vb_perc']:.2f}",
                    formatar_hausdorff(resultado_hib["hausdorff"]),
                    f"{resultado_hib['iou']:.2f}",
                    f"{resultado_hib['precision']:.2f}",
                    f"{resultado_hib['recall']:.2f}",
                    f"{resultado_hib['f1']:.2f}"
                ])

    print("-" * 70)
    print(f"[SUCESSO] Métricas salvas em: {CSV_SAIDA}")
    print(f"[SUCESSO] Imagens comparativas geradas em: {PASTA_PLOTS}")


if __name__ == "__main__":
    main()