
from sereTestLib.webservice.parameters_ws import  secret_password
from zipfile import ZipFile
from pathlib import Path

home = str(Path.home())



with ZipFile("seresensys.zip") as file:
    file.extractall(home + "/.ssh",pwd = secret_password)
