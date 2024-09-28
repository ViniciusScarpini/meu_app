from flask import Flask, request, render_template
from google.cloud import vision
import io
import os
import pandas as pd

app = Flask(__name__)

# Configuração do cliente da API do Google Cloud Vision
client = vision.ImageAnnotatorClient()

# Inicialização do DataFrame do Pandas
df_hidrometros = pd.DataFrame(columns=['imagem', 'numero'])

def detectar_numero_hidrometro(caminho_imagem):
    """Detecta texto na imagem usando Google Cloud Vision API."""
    with io.open(caminho_imagem, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts[0].description if texts else None

@app.route('/', methods=['GET', 'POST'])
def index():
    global df_hidrometros  # Para usar o DataFrame globalmente
    if request.method == 'POST':
        # Salvar arquivo enviado
        arquivo = request.files['imagem']
        caminho_imagem = os.path.join('static', arquivo.filename)
        arquivo.save(caminho_imagem)

        # Detectar número do hidrômetro
        numero_hidrometro = detectar_numero_hidrometro(caminho_imagem)

        # Adiciona a leitura ao DataFrame usando pd.concat
        nova_linha = pd.DataFrame({'imagem': [arquivo.filename], 'numero': [numero_hidrometro]})
        df_hidrometros = pd.concat([df_hidrometros, nova_linha], ignore_index=True)

        return render_template('resultados.html', numero=numero_hidrometro, imagem=arquivo.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
