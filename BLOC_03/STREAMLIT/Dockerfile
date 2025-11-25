FROM python:3.9-slim
WORKDIR /app

# Installe les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie requirements.txt et installe les dépendances Python
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copie TOUS les fichiers de l'application (dont app.py)
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD streamlit run --server.port 7860 app.py