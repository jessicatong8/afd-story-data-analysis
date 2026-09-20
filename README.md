# AFD Story Data Analysis

Streamlit application for cleaning Qualtrics survey exports, joining them with Supabase storybook metrics, reviewing completeness, and exporting the combined dataset.

## Development

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
streamlit run app.py
```

Copy `.env.example` to `.env` for local Supabase configuration, or add the values to Streamlit secrets when deployed. Uploaded survey data is processed in memory and is not written to disk.
