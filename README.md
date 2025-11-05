# HD Predictor (Streamlit)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run STREAMLIT/a.py
```

## Model Artifacts
Place the following files in the project root (same folder as this README):
- `huntington_model_pipeline.pkl`
- `target_encoder.pkl`
- `feature_encoders.pkl`
- `model_columns.json`

If these are missing, the app runs in Demo Mode with a simple heuristic predictor (for demonstration only; not medical advice).

## Demo Mode
When artifacts are not found, you will see a banner indicating Demo Mode. You can still use the Stage Prediction Tool; results are illustrative only.

## Pages
- Home: Overview with animated header
- About HD: Educational content
- Stage Prediction Tool: Input clinical values and get a stage prediction
- Resources: Curated links
- Wellness & Support Tips: Practical suggestions

## Notes
- Inputs align to UHDRS concepts. Use real clinical scales where appropriate.
- Batch upload and artifact path configuration can be added next.

