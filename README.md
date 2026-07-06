# LLM Network Model for Mental Health Analysis

This repository implements a Large Language Model (LLM)-based framework for topic modeling, network analysis, and an empathetic mental health chatbot. The project leverages modern NLP techniques to analyze online mental health discussions, uncover underlying themes, and support users through context-aware conversational AI.

## Project Overview

Online mental health forums contain rich, user-generated text that reflects real struggles, coping strategies, and emotional experiences. This project provides insights by combining:

- **Data Pipeline** to scrape, clean, and vectorize Beyond Blue forum data via a Dockerized Airflow ETL
- **Topic Modeling** to extract themes from mental health posts
- **Network Analysis** to visualize relationships between subtopics
- **Mental Health Chatbot** powered by Retrieval-Augmented Generation (RAG)

Together, these tools aim to support mental health research and explore AI-assisted early emotional support.

---
## Related Research Paper

This project builds on my earlier research on topic modeling and mental-health text analysis.
A detailed explanation of the methods and findings can be found in my published paper:

**Liao, Y.-C. (2023). _Subtopic Analysis of Mental Health Discussions Using BERTopic and Network Methods._
In AI Research and Development. Springer.**
Published version (subscription required): https://doi.org/10.1007/978-3-032-09832-0_11

Since the published version isn't openly accessible, a copy of the manuscript is included in this repo: [Mental_Health_Topic_Modelling_Paper.pdf](./Mental_Health_Topic_Modelling_Paper.pdf)

This paper supports the project's main objectives:

- **Objective 1 – Topic Modeling:**
  Provides the foundation for BERTopic experiments, tuning UMAP/HDBSCAN parameters,
  and refining labels using LLM-based review.

- **Objective 2 – Network Analysis:**
  Introduces the early development of subtopic co-occurrence networks and
  statistical validation methods, which are expanded in this project.
---

## Key Features

### 1. Data Pipeline (Docker + Airflow)
- Scrapes Beyond Blue forum posts with Selenium, orchestrated as a daily Airflow DAG
- Runs fully containerized (Docker Compose: Airflow + PostgreSQL), no local Python environment needed for collection
- Cleans text and converts emoji/emoticon/kaomoji into text equivalents to preserve emotional cues
- Stores raw cleaned posts as Parquet in Azure Blob Storage, with incremental (dedup-on-Post-ID) scraping
- Encodes posts into sentence embeddings and builds a FAISS index, stored back to Azure Blob Storage
- See [`Data_pipeline/README.md`](../Data_pipeline/README.md) for setup and architecture details

### 2. Topic Modeling
- Uses BERTopic with MentalBERT embeddings
- Extracts topics/subtopics for Depression, Anxiety, PTSD/Trauma, and Suicidal Thoughts
- Includes hyperparameter tuning for UMAP, HDBSCAN, and CountVectorizer
- Evaluated with topic coherence, and topic diversity
- Includes LLM-assisted topic refinement

### 3. Network Analysis
- Builds subtopic co-occurrence graphs
- Computes network metrics: modularity, centrality, and assortativity
- Produces interactive node-link visualizations
- Reveals how stressors (e.g., work, relationships, finances) influence multiple mental health conditions

### 4. Mental Health Chatbot
- Uses a RAG framework combining retrieval + LLM generation
- Integrates ICD-11 context (non-diagnostic, supportive framing only)
- Adapts tone using prompt engineering guidelines
- Supports multiple query modes: Original, Multi, and HyDE
- Includes sentiment analysis to improve emotional alignment
- Produces empathetic, safe, and context-aware responses

### 5. Data Preprocessing

**Data Collection (Beyond Blue, via Docker/Airflow):**
- Selenium-based scraping with incremental loads (skips posts already stored in Blob)
- Parses and normalizes post dates
- Converts kaomoji, emoticons, and Unicode emoji to text equivalents
- Output: cleaned Parquet files in Azure Blob Storage, partitioned by forum tag

**Topic Modeling Pipeline (downstream, in notebooks):**
- Removes irrelevant content using regular expressions
- Applies lowercasing, stop word removal, lemmatization, and tokenization
- Extracts bigrams and trigrams to capture domain-specific mental health phrases
- For BERT embeddings: minimal preprocessing (filtering, lowercasing) to preserve sentence context, since emoji/emoticon conversion already happened during collection

**RAG Database Pipeline:**
- Processes PDF guideline documents using pdfplumber
- Segments text into overlapping chunks (200 tokens, 40-token overlap) to maintain semantic continuity
- Cleans and normalizes text by removing non-linguistic artifacts, headers, footers, and formatting inconsistencies
- Anonymizes content to eliminate identifiable information from case examples
- Encodes chunks using MentalBERT-based sentence encoder for domain-specific embeddings
- Indexes vectors with FAISS for efficient similarity search

### 6. Human-in-the-Loop Evaluation
- Combines automated metrics with manual review
- Uses LLMs to refine topic labels and ensure semantic fit
- Ensures interpretability and quality of extracted topics

---

## Repository Structure

This project lives as two sibling top-level folders (opened together as one workspace):

```
Mental_Health_Analysis/
│
├── Data_pipeline/                  # Dockerized Airflow ETL (Beyond Blue collection)
│   ├── .vscode/
│   ├── dags/
│   │   └── orchestrator.py
│   ├── scripts/
│   │   ├── script1_scrape_clean.py
│   │   ├── script2_vectorise_save.py
│   │   └── kaomoji_to_text.json
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── README.md
│
└── LLM_NetworkMoel_MentalHealth/   # Topic modeling, network analysis, chatbot
    ├── data/
    ├── pic/
    ├── rag_system/
    ├── topic_modeling_result/
    ├── 1_topic_model.ipynb         # BERTopic subtopic extraction
    ├── 2_network_analysis.ipynb    # Subtopic co-occurrence network analysis
    ├── 3_RAG_chatbot.ipynb         # RAG chatbot (local + API modes)
    ├── hypertune.ipynb             # UMAP / HDBSCAN / CountVectorizer tuning
    ├── extracted_text.txt
    ├── kaomoji_to_text.json
    ├── requirement.txt             # Dependencies for notebooks / chatbot
    ├── .env
    └── README.md                   # this file
```

---

## Installation

This project has two independent environments: the **data pipeline** (Docker, for collection, in `Data_pipeline/`) and the **analysis & chatbot** stack (local Python, for everything downstream, in `LLM_NetworkMoel_MentalHealth/`).

### 1. Clone the repository

**Windows (Command Prompt or PowerShell):**
```cmd
git clone https://github.com/your-username/Mental_Health_Analysis.git
cd Mental_Health_Analysis
```

### 2. Set up the data pipeline (Docker)

```cmd
cd Data_pipeline
copy .env.example .env
```

Fill in `.env` with your Azure Blob Storage connection string and a generated Airflow Fernet key — see [`Data_pipeline/README.md`](../Data_pipeline/README.md) for full setup steps (including the `AIRFLOW_UID` and ChromeDriver notes).

```cmd
docker compose up -d --build
```

Open `http://localhost:8080` (airflow / airflow), enable the `mental_health_etl` DAG, and trigger it. This populates Azure Blob Storage with cleaned Beyond Blue posts and a FAISS index.

### 3. Install dependencies for analysis & chatbot

**Windows:**
```cmd
cd ../LLM_NetworkMoel_MentalHealth
pip install -r requirement.txt
```

### 4. Environment variable setup (analysis & chatbot)

Create a `.env` file inside `LLM_NetworkMoel_MentalHealth/` and add your API credentials:

```dotenv
HF_TOKEN=your_huggingface_token
ICD_CLIENT_ID=your_icd_client_id
ICD_CLIENT_SECRET=your_icd_client_secret
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
AZURE_CONTAINER_NAME=mh-etl-data
```

**Variable descriptions:**
- `HF_TOKEN`: Hugging Face authentication token for accessing MentalBERT and other models
- `ICD_CLIENT_ID`, `ICD_CLIENT_SECRET`: ICD-11 API credentials for retrieving clinical guideline data
- `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_CONTAINER_NAME`: used to read the Parquet/FAISS artifacts produced by the data pipeline

**How to obtain credentials:**
- **Hugging Face**: Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **ICD-11 API**: Request access at [icd.who.int/icdapi](https://icd.who.int/icdapi)

### 5. Download model files

The **MentalBERT** model will be automatically downloaded from Hugging Face when first running the scripts. Ensure your `HF_TOKEN` is configured if accessing gated models.

---

## Usage

### 1. Data Collection
- Run the Docker/Airflow pipeline in `Data_pipeline/` (see Installation step 2) to scrape and clean Beyond Blue forum posts and build the FAISS index. Output lands in Azure Blob Storage — no notebook step required for collection.

### 2. Topic Modeling
**Generate subtopics:**
- Run `1_topic_model.ipynb` to extract subtopics using BERTopic with MentalBERT embeddings, reading the cleaned Parquet data from Azure Blob Storage

**Optimize hyperparameters:**
- Run `hypertune.ipynb` to fine-tune UMAP, HDBSCAN, and CountVectorizer parameters for improved topic quality and coherence

### 3. Network Analysis
- Run `2_network_analysis.ipynb` to build and visualize subtopic co-occurrence networks, revealing relationships and connections across mental health themes.

### 4. Mental Health Chatbot
- Run `3_RAG_chatbot.ipynb` — covers both local and API-backed modes, plus the Original / Multi / HyDE query strategies, within the notebook

---

## Results

### Topic Modeling
- Extracted meaningful themes across four mental health conditions: Depression, Anxiety, PTSD, and Suicidal Thoughts
- Validated topic quality using coherence and diversity metrics
- Enhanced topic labels through human review and LLM-assisted refinement for improved interpretability

### Network Analysis
- Identified central subtopics including coping strategies, medication side effects, and family issues
- Revealed cross-condition connections showing how different mental health themes intersect
- Generated visual node-link diagrams highlighting key patterns and relationships between subtopics
- Conducted hypothesis testing to validate network structure and community detection

### Mental Health Chatbot
- Generates empathetic, clear, and emotionally appropriate responses
- Provides ICD-11 contextual information with supportive framing (non-diagnostic)
- Dynamically adapts conversational tone based on user emotional state
- Integrates RAG pipeline for evidence-based mental health support

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for any purpose, including commercial applications, as long as you include the original copyright notice.

---

## Acknowledgments

- **Dr. Indu Bala** - Project supervisor and advisor
- **meta-llama/Llama-3.1-405B-Instruct** - Foundation model for the mental health chatbot
- **MentalBERT** - Domain-specific embeddings for mental health text
- **BERTopic** - Topic modeling framework
- **Beyond Blue** - Mental health forum data source
- **ICD-11 API** - Clinical guideline and contextual mental health resources
- **Apache Airflow / Docker** - Orchestration and containerization for the data collection pipeline