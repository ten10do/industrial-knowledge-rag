# Archived Streamlit prototype

This directory preserves the original Streamlit proof of concept for historical reference. It is not part of the supported FastAPI/React runtime, release image, CI quality gate, or production security boundary.

Use the repository root `README.md` for the supported application. If the prototype must be inspected locally, install its exact lock file in an isolated environment and run it from this directory:

```powershell
python -m pip install -r legacy_streamlit\requirements.txt
streamlit run legacy_streamlit\app.py
```

The prototype intentionally retains `langchain-community` because it is archived rather than maintained. Do not deploy it or mix its environment with the supported backend.
