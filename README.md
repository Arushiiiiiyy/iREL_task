## 1) Methodology (What approach is used)

### Pipeline design
1. **Source registry** (`config/videos.yaml`)
   - Stores 5 selected videos, links, domain, and declared code-mixed language.
2. **Ingestion** (`video_ingest.py`)
   - Downloads audio via `yt-dlp`.
3. **ASR** (`asr.py`)
   - Uses `faster-whisper` to generate transcript segments with timestamps.
4. **Preprocessing + language tagging** (`preprocess.py`)
   - Cleans text and detects language (segment-level fallback).
5. **Concept extraction** (`concept_extractor.py`)
   - Extracts technical concepts from segment text (regex starter + normalization).
6. **Linguistic standardization** (`standardizer.py`)
   - Maps colloquial/code-mixed terms to standard academic English terms.
7. **Prerequisite mapping** (`prereq_mapper.py`)
   - Builds edges based on pedagogical order and dependency cue phrases.
8. **Graph representation** (`graph_builder.py`)
   - Exports a strict JSON DAG-style format: nodes, edges, normalization map, mentions, metadata.

### Why this method
- Handles **code-mixed/noisy** content without requiring perfect grammar.
- Keeps a **transparent and auditable** mapping from raw phrases to normalized terms.
- Produces a format that is easy to analyze programmatically and visualize later in D3/Neo4j/NetworkX.

---

## 2) Source disclosure (5 videos) + language declaration

Defined in `config/videos.yaml`.

| ID | Domain | Source Link | Declared Code-Mix |
|---|---|---|---|
| `v1_hinglish_dsa` | Computer Science | https://www.youtube.com/watch?v=RBSGKlAvoiM | Hindi-English |
| `v2_hindi_os` | Computer Science | https://www.youtube.com/watch?v=vBURTt97EkA | Hindi-English |
| `v3_telugu_java` | Computer Science | https://www.youtube.com/watch?v=eIrMbAQSU34 | Telugu-English |
| `v4_tamil_networking` | Computer Science | https://www.youtube.com/watch?v=qiQR5rTSshw | Tamil-English |
| `v5_hinglish_physics` | Physics | https://www.youtube.com/watch?v=Y4jYxk2M4kE | Hindi-English |

> Note: If any link is region-blocked or unavailable at runtime, replace that entry with another public code-mixed technical lecture.

---

## 3) Output schema (strict machine-readable format)

Each video generates:

`outputs/<video_id>.knowledge_graph.json`

Schema:
- `meta`
  - `video_id`, `title`, `url`, `domain`, `languages_detected`
- `nodes`
  - list of concept nodes (`{"id": "..."}`)
- `edges`
  - prerequisite directed edges:
  - `source`, `target`, `reason`, `confidence`
- `lexical_normalization_map`
  - raw colloquial/code-mixed phrase -> standard term
- `concept_mentions`
  - mention-level evidence (`segment_index`, confidence)

---

## 4) Project structure

```text
.
├── config/
│   └── videos.yaml
├── data/                      # generated transcripts/audio per video
├── outputs/                   # generated JSON concept graphs
├── scripts/
│   └── run_pipeline.sh
├── src/cmpfe/
│   ├── __init__.py
│   ├── asr.py
│   ├── cli.py
│   ├── concept_extractor.py
│   ├── graph_builder.py
│   ├── io_utils.py
│   ├── models.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── prereq_mapper.py
│   ├── standardizer.py
│   └── video_ingest.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 5) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Also ensure `ffmpeg` is installed on your machine (needed by `yt-dlp --extract-audio` and Whisper audio loading).

On macOS with Homebrew:

```bash
brew install ffmpeg
```

---

## 6) Run

### Full run (download + ASR + extraction)
```bash
scripts/run_pipeline.sh
```

If Whisper model download fails with corporate or self-signed TLS interception, provide your trusted CA bundle:

```bash
scripts/run_pipeline.sh --ssl-ca-file /path/to/company-ca.pem
```

Or set it once through the environment:

```bash
export CMPFE_SSL_CA_FILE=/path/to/company-ca.pem
scripts/run_pipeline.sh
```

As a last resort for isolated environments you control, you can disable verification only for the Whisper model download step:

```bash
scripts/run_pipeline.sh --insecure-model-download
```

### If transcripts already exist (`data/<video_id>/transcript.json`)
```bash
scripts/run_pipeline.sh --skip-download
```

---

## 7) Demonstration video requirement (for final submission)

For hackathon submission you must include:
1. Public GitHub repository link.
2. Demo video link showing pipeline execution + resulting JSON graph.

Add your demo link here once recorded:

`Demo: <ADD_YOUR_UNLISTED_YOUTUBE_OR_DRIVE_LINK>`

---

## 8) Suggested improvements (for higher evaluation score)

- Replace regex extractor with LLM-based concept chunking + ontology linking.
- Add multilingual transliteration normalization for Hindi/Telugu/Tamil scripts.
- Add edge confidence calibration from multiple signals (order + cue + semantic entailment).
- Add graph visualization notebook (PyVis/Plotly/D3 export).
- Add benchmark set + manual annotations for relational accuracy.


recording link

https://drive.google.com/file/d/1hyE6dj5NjeLQXe_eUV_QjIbN0T0CuqL7/view?usp=sharing
