# Book site: Context Engineering with DSPy

A static site for the book, ready for Cloudflare Pages. Everything served is in `public/`; there is no build step on Cloudflare.

- `/` is the landing page: cover, buy links, chapter list, the eight steps, Mike's DSPy talks, articles and code, and an FAQ with structured data.
- `/dspy-in-eight-steps/` is Chapter 3 in full. It is unlisted: nothing links to it, it is left out of the sitemap, and it is marked `noindex`. Share the URL directly.

## Deploy

The site deploys as a Cloudflare Worker serving static assets, configured in `wrangler.toml`.

**From the dashboard:** Workers & Pages → Create → import the `hammer-mt/context-engineering-dspy-book` repository. Leave the build command empty, set the deploy command to `npx wrangler deploy`, and under advanced settings set the root directory to `website` and pick the branch to deploy.

The Workers build settings that work: production branch set to the branch holding `website/`, build command empty, deploy command `npx wrangler deploy`, root directory `/website`. Every push to that branch redeploys.

**From the command line:**

```bash
cd website && npx wrangler deploy
```

## Change things

Edit `src/index.html` (landing page), `public/styles.css`, or the settings and FAQ at the top of `build.py`, then rebuild:

```bash
pip install markdown pygments
```

```bash
python3 website/build.py
```

Settings in `build.py`:

- `SITE`: the site's address, used for canonical links, the sitemap and social cards. Set it to the real domain before launch.
- `AMAZON`: the Amazon product link.
- `CHAPTER_INDEXABLE`: set to `True` to let search engines index the free chapter and add it to the sitemap.

The chapter is rendered from `content/dspy-in-eight-steps.md`. Figures are not included because the Markdown conversion kept only their captions.

`keywords.md` holds the search-volume research the page copy and FAQ were written around.
