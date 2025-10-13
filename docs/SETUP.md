# GitHub Pages Setup

## Quick Setup

1. **Enable GitHub Pages**:

   - Go to repo **Settings** → **Pages**
   - Source: "Deploy from a branch"
   - Branch: "main", Folder: "/ (docs)"
   - Click **Save**

2. **Visit your site** (after 1-2 min):

   ```
   https://<username>.github.io/<repository>/
   ```

3. **Update repo info** (optional):
   - Click ⚙️ next to "About" on repo homepage
   - Check "Use your GitHub Pages website"

## Regenerate Visualizations

When you add new results:

```bash
pip install plotly
python scripts/generate_interactive_plot.py
python scripts/generate_svg_preview.py
```

Or just push - the GitHub Action auto-regenerates!

## Local Testing

```bash
python -m http.server 8000 --directory docs
# Visit http://localhost:8000
```
