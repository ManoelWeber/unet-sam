import cv2
import numpy as np
import os

# Lista para armazenar os pontos do contorno atual
pontos = []
imagem_aux = None

def capturar_cliques(event, x, y, flags, param):
    global pontos, imagem_aux
    
    # Clique com botão esquerdo adiciona um ponto ao contorno
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos.append((x, y))
        # Desenha um círculo no local do clique para feedback visual
        cv2.circle(imagem_aux, (x, y), 2, (0, 255, 0), -1)
        
        # Se houver mais de um ponto, desenha a linha conectando o anterior ao atual
        if len(pontos) > 1:
            cv2.line(imagem_aux, pontos[-2], pontos[-1], (0, 255, 0), 1)
        
        cv2.imshow("Anotacao de Desgaste - UFSC", imagem_aux)

def anotar_imagem(caminho_imagem, pasta_destino):
    global pontos, imagem_aux
    pontos = [] # Reseta os pontos para a nova imagem
    
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"[ERRO] Não foi possível carregar: {caminho_imagem}")
        return True # Pula para a próxima imagem

    imagem_aux = img.copy()
    nome_arquivo = os.path.basename(caminho_imagem)
    nome_puro = os.path.splitext(nome_arquivo)[0]
    
    cv2.namedWindow("Anotacao de Desgaste - UFSC")
    cv2.setMouseCallback("Anotacao de Desgaste - UFSC", capturar_cliques)
    
    print(f"\n==================================================")
    print(f"ANOTANDO: {nome_arquivo}")
    print(f"==================================================")
    print("-> Clique com o Botão Esquerdo para delimitar o contorno.")
    print("-> Pressione 'c' para FECHAR o polígono e SALVAR a máscara.")
    print("-> Pressione 'r' para REINICIAR o desenho desta imagem.")
    print("-> Pressione 's' para PULAR esta imagem sem salvar.")
    print("-> Pressione 'q' para SAIR do programa.")

    while True:
        cv2.imshow("Anotacao de Desgaste - UFSC", imagem_aux)
        key = cv2.waitKey(1) & 0xFF
        
        # 'r' ou 'R' - Reiniciar o desenho
        if key in [ord('r'), ord('R')]:
            pontos = []
            imagem_aux = img.copy()
            print("Desenho reiniciado.")
            
        # 's' ou 'S' - Pular imagem
        elif key in [ord('s'), ord('S')]:
            print(f"Imagem {nome_arquivo} pulada pelo usuário.")
            break

        # 'c' ou 'C' - Concluir, fechar polígono e salvar (Correção do Caps Lock)
        elif key in [ord('c'), ord('C')]:
            if len(pontos) > 2:
                # Criar uma máscara preta com o mesmo tamanho da imagem original
                mascara = np.zeros(img.shape[:2], dtype=np.uint8)
                
                # Converter a lista de pontos para o formato do OpenCV
                pts_array = np.array(pontos, dtype=np.int32).reshape((-1, 1, 2))
                
                # Preencher o polígono desenhado com a cor branca (255)
                cv2.fillPoly(mascara, [pts_array], 255)
                
                # Desenhar a linha de fechamento na tela de visualização para feedback
                cv2.line(imagem_aux, pontos[-1], pontos[0], (0, 255, 0), 1)
                cv2.imshow("Anotacao de Desgaste - UFSC", imagem_aux)
                cv2.waitKey(300) 
                
                # Garante que a pasta destino existe
                os.makedirs(pasta_destino, exist_ok=True)
                
                # Salva a máscara e checa se o OpenCV realmente conseguiu escrever o arquivo
                caminho_salvamento = os.path.join(pasta_destino, f"{nome_puro}_mask.png")
                sucesso = cv2.imwrite(caminho_salvamento, mascara)
                
                if sucesso:
                    print(f"[SUCESSO] Máscara salva com sucesso em:\n -> {caminho_salvamento}")
                else:
                    print(f"[ERRO CRÍTICO] O OpenCV falhou ao tentar salvar em:\n -> {caminho_salvamento}\nVerifique as permissões da pasta!")
                break
            else:
                print("[AVISO] Adicione pelo menos 3 pontos para fechar o polígono do desgaste!")
                
        # 'q' ou 'Q' - Sair do script completamente
        elif key in [ord('q'), ord('Q')]:
            print("\nProcesso interrompido pelo usuário. Fechando...")
            return False

    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    diretorio_base = r"C:\Projetos\SAM_V2"
    subpastas_origem = ["train", "test"]
    
    # ATENÇÃO: O código vai criar e salvar as imagens nesta pasta abaixo
    diretorio_saida_pai = os.path.join(diretorio_base, "Anotacao Manual")
    
    extensoes_suportadas = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    execucao_ativa = True

    print("Inicializando Pipeline de Anotação Manual para o TCC...")
    
    for subpasta in subpastas_origem:
        if not execucao_ativa:
            break
            
        pasta_imagens = os.path.join(diretorio_base, subpasta)
        pasta_destino_mascaras = os.path.join(diretorio_saida_pai, subpasta)
        
        if not os.path.exists(pasta_imagens):
            print(f"[AVISO] Pasta de origem não encontrada, pulando: {pasta_imagens}")
            continue
            
        print(f"\n>>> Processando a pasta: {subpasta.upper()} <<<")
        
        arquivos = [f for f in os.listdir(pasta_imagens) if f.lower().endswith(extensoes_suportadas)]
        
        if not arquivos:
            print(f"Nenhuma imagem encontrada em {pasta_imagens}")
            continue
            
        for arquivo in arquivos:
            caminho_completo = os.path.join(pasta_imagens, arquivo)
            
            execucao_ativa = anotar_imagem(caminho_completo, pasta_destino_mascaras)
            
            if not execucao_ativa:
                break

    print("\nScript finalizado.")