# AI Customer Support Copilot

## Segment 1 - Data Foundation

This segment creates a deterministic 20,000-row synthetic customer-support dataset strictly from the ZENDS fact catalog in `1_Data_Foundation/config/zends_facts.json`. The catalog is a hand-checked transcription of `ZENDS Communications.pdf`; the pre-existing example CSV and notebook are not read by this pipeline.

### Outputs

- `1_Data_Foundation/data/zends_support_queries_20000.csv` - required `text`, `intent`, and `sentiment` columns.
- `1_Data_Foundation/data/train.csv`, `validation.csv`, `test.csv` - 70/15/15, jointly stratified by intent and sentiment.
- `1_Data_Foundation/evaluation/generation_manifest.json` - source and reproducibility metadata.
- `1_Data_Foundation/evaluation/validation_report.json` - quality and factual-token checks.
- `1_Data_Foundation/evaluation/eda/` - requested charts and EDA summary.

### Run

Install the Segment 1 dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_segment_1.py
python -m pytest -q
```

The run is deterministic with seed `20260911`. Segment 1 deliberately stops after data generation, validation, EDA, and splitting. No NLP, RAG, LLM, or Streamlit code is included yet.

## Segment 5 - Streamlit Support Workspace

The Streamlit application is a light-theme enterprise support workspace that reuses the existing Segment 4 `ZendsResponseEngine`. It does not duplicate the NLP, retrieval, or grounding logic.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run 5_Streamlit_Integration/app.py
```

The application displays the existing intent, sentiment, priority, retrieved ZENDS source references, and grounded recommended response. It also surfaces Segment 4 abstentions when the retrieved context cannot support a company-specific answer. See [Segment 5 documentation](5_Streamlit_Integration/README.md) for deployment preparation and limitations.
