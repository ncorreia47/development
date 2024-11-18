"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import time
import google.generativeai as genai
from dotenv import load_dotenv as env

env(dotenv_path='variables.env')
genai.configure(api_key=os.getenv("API_KEY_GEMINI_TEST"))


def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("insurance.csv", mime_type="text/csv"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
        "How many genders have in this dataset?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "There are **2** genders in this dataset: **female** and **male**. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Qual o valor médio por sexo?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```python\nimport pandas as pd\n\n# Carregando os dados do arquivo CSV\ndata = pd.read_csv(\"insurance.csv\")\n\n# Calculando o valor médio das cobranças por sexo\naverage_charges_by_sex = data.groupby(\"sex\")[\"charges\"].mean()\n\n# Exibindo os resultados\nprint(average_charges_by_sex)\n```\n\nIsso imprimirá o valor médio das cobranças para cada sexo:\n\n```\nsex\nfemale    12569.578844\nmale      13956.751178\nName: charges, dtype: float64\n```\n\nPortanto, o valor médio das cobranças para mulheres é **12569.58**, enquanto para homens é **13956.75**. \n",
      ],
    },
  ]
)


response = chat_session.send_message("Qual a soma total da coluna charges?")

print(response.text)