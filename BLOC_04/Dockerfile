# ========================
# 1️⃣ Image de base Python légère
# ========================
FROM python:3.10-slim

# ========================
# 2️⃣ Variables d’environnement
# ========================
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

# ========================
# 3️⃣ Installation minimale des dépendances système
# ========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ========================
# 4️⃣ Copie du code et des fichiers de config
# ========================
COPY dags/ dags/
COPY plugins/ plugins/
COPY tests/ tests/
COPY requirements.txt .
COPY pytest.ini .

# ========================
# 5️⃣ Installation des dépendances Python
# ========================
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pytest pytest-html pytest-mock

# ========================
# 6️⃣ Commande par défaut : exécuter les tests
# ========================
CMD ["pytest", "-v", "--maxfail=1", "--disable-warnings", "--junitxml=report.xml", "--html=report.html", "--self-contained-html"]
