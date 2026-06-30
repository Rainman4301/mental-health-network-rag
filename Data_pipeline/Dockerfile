# ---------------------------------------------------------------------------
# Extend the official Airflow image with all packages the ETL scripts need.
# We install as root first (system deps), then switch back to airflow user.
# ---------------------------------------------------------------------------
FROM apache/airflow:2.10.0

USER root

# Chrome + ChromeDriver for Selenium headless scraping
# Chrome + ChromeDriver for Selenium headless scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \
    curl \
    ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
       | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Python packages used by both ETL scripts
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt