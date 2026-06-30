# ---------------------------------------------------------------------------
# Extend the official Airflow image with all packages the ETL scripts need.
# We install as root first (system deps), then switch back to airflow user.
# ---------------------------------------------------------------------------
FROM apache/airflow:2.10.0

USER root

# Chrome + ChromeDriver for Selenium headless scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \
    curl \
    jq \
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

# ---------------------------------------------------------------------------
# Pin a ChromeDriver build that EXACTLY matches the Chrome version we just
# installed above, downloaded once at build time from the official
# Chrome for Testing JSON API. This means:
#   - no network calls at DAG runtime to resolve/download a driver
#   - no dependency on writable $HOME / cache dirs inside the container
#   - the driver version always matches Chrome, because it's derived from
#     whatever Chrome version this build just installed
# Re-running `docker compose up --build` re-resolves both together.
# ---------------------------------------------------------------------------
RUN CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+') \
    && echo "Installed Chrome version: ${CHROME_VERSION}" \
    && CHROMEDRIVER_URL=$(curl -sS "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json" \
       | jq -r --arg V "$CHROME_VERSION" \
         '.versions[] | select(.version == $V) | .downloads.chromedriver[]? | select(.platform == "linux64") | .url') \
    && if [ -z "$CHROMEDRIVER_URL" ]; then \
         echo "No exact chromedriver match for ${CHROME_VERSION} — falling back to the latest build for the same major version" ; \
         MAJOR=$(echo "$CHROME_VERSION" | cut -d. -f1) ; \
         CHROMEDRIVER_URL=$(curl -sS "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json" \
            | jq -r --arg M "${MAJOR}." \
              '[.versions[] | select(.version | startswith($M))][-1].downloads.chromedriver[]? | select(.platform == "linux64") | .url') ; \
       fi \
    && echo "ChromeDriver download URL: ${CHROMEDRIVER_URL}" \
    && test -n "$CHROMEDRIVER_URL" \
    && wget -q -O /tmp/chromedriver.zip "$CHROMEDRIVER_URL" \
    && unzip -j /tmp/chromedriver.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/chromedriver \
    && rm /tmp/chromedriver.zip \
    && /usr/local/bin/chromedriver --version

# Available to scripts at runtime — script1 reads this directly, no
# webdriver-manager / runtime resolution needed.
ENV CHROMEDRIVER_PATH=/usr/local/bin/chromedriver

USER airflow

# Python packages used by both ETL scripts
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt