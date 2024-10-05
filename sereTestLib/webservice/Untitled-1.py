#%%
import requests
import json

BASE_URL = 'http://192.9.200.90:5004/'
cedula="49"
print(cedula)
print(f"{BASE_URL}id/{cedula}")
response = requests.get(f"{BASE_URL}id/{cedula}") #esta respuesta va a ser un diccionario


json_r=json.loads(response.text)
print(json_r)
print(json_r['result'])
print(type(json_r))
dicts = json.loads(json_r['IDENTIFICACION'])
print(dicts['IDENTIFICACION'])
# %%
