import pandas as pd
from google.cloud import dialogflow_v2beta1 as dialogflow
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Defina a chave de API do Google Cloud (substitua pelo seu caminho da chave JSON)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "caminho/para/sua-chave.json"

# Exemplo de DataFrame com dados fictícios
data = {
    "Produto": ["Camiseta", "Calça", "Jaqueta", "Tênis"],
    "Preço": [49.90, 89.90, 159.90, 120.00],
    "Disponível": [True, False, True, True]
}

df = pd.DataFrame(data)

# Função para buscar informações no DataFrame com base na pergunta
def buscar_no_dataframe(pergunta: str):
    if "preço" in pergunta.lower():
        # Pergunta sobre preço de produtos
        produto = pergunta.lower().split("preço de")[-1].strip()
        resultado = df[df['Produto'].str.contains(produto, case=False)]
        if not resultado.empty:
            return f'O preço da {produto} é R${resultado["Preço"].values[0]:.2f}.'
        else:
            return "Produto não encontrado."
    elif "disponível" in pergunta.lower():
        # Pergunta sobre disponibilidade de produtos
        produto = pergunta.lower().split("disponível")[-1].strip()
        resultado = df[df['Produto'].str.contains(produto, case=False)]
        if not resultado.empty:
            disponivel = "disponível" if resultado["Disponível"].values[0] else "não disponível"
            return f'A {produto} está {disponivel}.'
        else:
            return "Produto não encontrado."
    else:
        return "Desculpe, não entendi a sua pergunta."

# Função para interagir com o Gemini usando a API do Dialogflow (supondo que você tenha configurado o Dialogflow)
def chamar_gemini(pergunta: str):
    # Instanciando o cliente do Dialogflow
    client = dialogflow.SessionsClient()

    # Criação de uma sessão com um ID único (pode ser qualquer string)
    session = client.session_path('seu-projeto-id', 'unique-session-id')

    # Criando o texto da consulta (pergunta)
    text_input = dialogflow.TextInput(text=pergunta, language_code="pt-BR")
    query_input = dialogflow.QueryInput(text=text_input)

    # Enviando a consulta para o modelo Gemini via Dialogflow
    response = client.detect_intent(request={"session": session, "query_input": query_input})

    # Obtendo a resposta do Gemini
    return response.query_result.fulfillment_text

# Configurando o modelo Gemini via LangChain (por exemplo, usando um agente)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Definindo o template do prompt
prompt_template = """
Você é um assistente de vendas de uma loja online. As informações sobre os produtos estão organizadas em um DataFrame.

Aqui estão alguns produtos e seus detalhes:

{tabela}

Responda à seguinte pergunta com base nessas informações: {pergunta}
"""

# Criando o PromptTemplate
prompt = PromptTemplate(input_variables=["tabela", "pergunta"], template=prompt_template)

# Criando o "chain" com o modelo Gemini (usando o dialogflow para interagir)
chain = LLMChain(llm=chamar_gemini, prompt=prompt)

# Função para usar LangChain com base no DataFrame
def responder_com_dataframe(pergunta: str):
    # Converte o DataFrame para uma string (em formato de tabela) para passar ao modelo
    tabela = df.to_string(index=False)
    
    # Faz a consulta para obter a resposta
    resposta = chain.run(tabela=tabela, pergunta=pergunta)
    return resposta

# Função para buscar a resposta diretamente
def responder_pergunta(pergunta: str):
    # Primeiro, tenta responder com base nos dados do DataFrame
    resposta_dataframe = buscar_no_dataframe(pergunta)
    
    # Se não souber responder, chama o Gemini
    if "Desculpe" in resposta_dataframe:
        return responder_com_dataframe(pergunta)
    return resposta_dataframe

# Exemplo de uso
if __name__ == "__main__":
    pergunta = input("Qual é a sua pergunta? ")
    resposta = responder_pergunta(pergunta)
    print(f"Resposta: {resposta}")
