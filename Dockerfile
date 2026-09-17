# Stage 1: FastApi Application Context
FROM python:3.12-slim

WORKDIR /app/src

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependencies
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MS ODBC Driver 18
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy contents of local src folder directly into /app/src
COPY src_mcp /app/src_mcp
COPY mcp /app/mcp
RUN mkdir -p /app/logs

EXPOSE 8050
EXPOSE 9999

# Run app directly from working directory /app/src
CMD ["uvicorn", "src_mcp.app:app", "--host", "0.0.0.0", "--port", "9999"]