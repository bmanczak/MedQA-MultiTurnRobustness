# GitHub Pages Setup Guide

This guide explains how to enable and configure GitHub Pages for the interactive visualization.

## Quick Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (in the left sidebar)
3. Under **Source**, select:
   - Source: **Deploy from a branch**
   - Branch: **main** (or **master**)
   - Folder: **/ (docs)**
4. Click **Save**

GitHub will automatically build and deploy your site. It will be available at:

```
https://<username>.github.io/<repository-name>/
```

For this repository:

```
https://donttrustmedicalais.github.io/medqa_deep_robustness/
```

### 2. Update README Links (if needed)

If your GitHub username or repository name differs, update the links in `README.md`:

```markdown
https://donttrustmedicalais.github.io/medqa_deep_robustness/
```

Replace with your actual GitHub Pages URL.

### 3. Set Website in Repository Info

1. Go to the main page of your repository
2. Click the **⚙️ gear icon** next to "About" (top-right)
3. Check **Use your GitHub Pages website**
4. Add topics like: `medical-ai`, `llm-robustness`, `benchmark`, `visualization`
5. Click **Save changes**

This makes the live demo discoverable right from the repository homepage.

## File Structure

```
docs/
├── index.html          # Landing page with styled iframe
├── chart.html          # Interactive Plotly chart (auto-generated)
├── preview.svg         # Static preview for README (auto-generated)
├── .nojekyll          # Tells GitHub Pages not to use Jekyll
└── SETUP.md           # This file
```

## Automatic Updates

The repository includes a GitHub Action (`.github/workflows/update-preview.yml`) that automatically regenerates the visualization when:

- New results are added to `results/`
- Visualization scripts are modified
- Manually triggered via "Actions" tab

## Troubleshooting

### Pages not loading?

- Ensure the `docs/` folder is committed and pushed to the main branch
- Check the **Actions** tab for any deployment failures
- Verify GitHub Pages is enabled in Settings → Pages

### Chart not displaying?

- Check browser console for errors
- Verify `chart.html` exists in `docs/` folder
- Ensure the iframe src in `index.html` is correct

### Preview image not showing in README?

- Verify `docs/preview.svg` exists and is committed
- Check the image path in README matches: `docs/preview.svg`
- Note: SVG images may not render in some email clients (but work fine on GitHub)

## Customization

### Styling

Edit `docs/index.html` to customize:

- Header colors and fonts
- Instructions text
- Layout and spacing
- Mobile responsiveness

### Chart Configuration

Modify `scripts/generate_interactive_plot.py` to adjust:

- Color palette
- Model name mappings
- Chart dimensions
- Default interventions shown

## Local Testing

To test the GitHub Pages site locally:

```bash
# Install a simple HTTP server
python -m http.server 8000 --directory docs

# Open in browser
open http://localhost:8000
```

Or use any static file server.

## Performance

- The `chart.html` file is large (~15-20 MB) because it's a self-contained Plotly export
- It loads quickly on GitHub Pages due to their CDN
- First load may take a few seconds; subsequent loads are cached

## Security

- All files are static HTML/JS/SVG
- No server-side processing
- No user data collection
- Safe to host on public GitHub Pages
