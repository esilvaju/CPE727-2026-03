FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MPLBACKEND=Agg

# Copia e instala os requisitos
COPY Seminarios/regularizacao/exemplos/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto dos arquivos
COPY . .

CMD ["python", "Seminarios/regularizacao/exemplos/executaExemplos.py"]
