import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Garantir que a saída do terminal seja compatível com UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Carregar o modelo pré-treinado VGG16
model = VGG16(weights='imagenet')

# Função para carregar e processar a imagem
def prepare_image(img_path):
    # Carregar a imagem com o tamanho esperado pelo modelo (224x224)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Converter para array NumPy
    img_array = image.img_to_array(img)
    
    # Adicionar uma dimensão extra para que a imagem tenha a forma (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Pré-processar a imagem (normalizar conforme o esperado pelo modelo)
    img_array = preprocess_input(img_array)
    
    return img_array

# Função para fazer a previsão
def predict_image(img_path):
    print('Preparing image...')
    img_array = prepare_image(img_path)
    
    # Fazer a previsão usando o modelo
    print('Doing first predictions...')
    predictions = model.predict(img_array)
    
    # Decodificar a previsão em um formato legível
    print('Decoding predictions...')
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    # Exibir as predições com codificação UTF-8
    print("Top 3 predições:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}. {label}: {score:.2f}")
    
    return decoded_predictions

# Função para exibir a imagem
def show_image(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Caminho da imagem que você deseja analisar
img_path = 'ia-projects/sample_image/polar_bear.jpg'  # Substitua pelo caminho correto da imagem

# Prever o conteúdo da imagem
predict_image(img_path)
