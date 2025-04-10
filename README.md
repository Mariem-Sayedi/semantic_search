**1. Cloner le projet **


cd fastapi-project

**2. Créer un environnement virtuel **

python3 -m venv venv
source venv/bin/activate #sur wsl ubuntu


**3. Installer les dépendances **

pip install -r requirements.txt

** Ecécuter 
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz
!gunzip cc.fr.300.bin.gz


**4. Lancer le serveur **
uvicorn main:app --reload


**. Tester dans swagger **
http://127.0.0.1:8000/docs
