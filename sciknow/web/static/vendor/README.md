# Vendored frontend libraries

These files are served by the FastAPI app at `/static/vendor/` to avoid
the external jsDelivr CDN dependency. See `sciknow/web/app.py` for the
mount + the HTML tags that reference them.

## Versions + origin

| File(s) | Version | Origin |
|---|---|---|
| `katex/katex.min.css` | 0.16.11 | https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css |
| `katex/katex.min.js` | 0.16.11 | https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js |
| `katex/contrib/auto-render.min.js` | 0.16.11 | https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js |
| `katex/fonts/*.woff2` (20 files) | 0.16.11 | https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/ |
| `echarts/echarts.min.js` | 5.5.1 | https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js |

Total footprint: ~1.8 MB. Committed to the repo so `uv sync` +
`sciknow book serve` works offline.

## Licenses

- KaTeX — [MIT](https://github.com/KaTeX/KaTeX/blob/main/LICENSE), © KaTeX contributors.
- ECharts — [Apache 2.0](https://github.com/apache/echarts/blob/master/LICENSE), © Apache Software Foundation.

## Refreshing

Bump the URLs above to the new release, re-download, update the version
column, run L1 + a manual browser smoke on any wiki page with `$...$`
math + the Topic Map viz tab. Keep the file paths stable so the HTML
script tags don't need churn.
