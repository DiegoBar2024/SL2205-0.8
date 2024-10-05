# SL2205
Repositorio del proyecto SereTest


## Instrucciones para instalar el ambiente de prueba

### Paso 1: Instalar docker


- Para Ubuntu: Seguir los pasos del link: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository y https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user

- Para Windows: Seguir los pasos del link: https://docs.docker.com/desktop/install/windows-install/

- Instalar la extensión de Docker en VSCode.

### Paso 2: Clonar el repositorio Sere_dockers

- git clone git@github.com:SereLabs/Sere_dockers.git

- git checkout mvt_sl2205_test

### Paso 3: Levantar los dockers

- Para Windows, cambiar en el compose donde dice HOME cambiar a HOME_WINDOWS (verificar que el parámetro  en el archivo .env coincida con el home de su PC). Se puede cambiar el path de los volumes, tener en cuenta que el volume de inferencia debe tener los archivos necesarios simulando los que están en home/sere/Dropbox/PROJECTS/SL2205/sereDataTesting

- docker compose -p sl2205-docker-testing build --no-cache

- docker compose -p sl2205-docker-testing create


### Paso 4: Acceder a los Web Services

- Para acceder a los webservices se ingresa a la ip 127.0.0.1 puerto 3000.

- Se puede verificar en http://127.0.0.1:3000/docs


