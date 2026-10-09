---
name: landing-page
description: Build a single-file landing page (index.html with inline CSS) from a product brief and a brand guide. Use when the user asks for a landing page.
---

# Landing page

Build a landing page from the brief and the brand guide you are given.

## Output

- Return one complete `index.html` document and nothing else: no explanation and no Markdown code fences.
- Put all CSS in a single `<style>` element. Use minimal JavaScript, or none. No frameworks, no build step, no external stylesheets or scripts.
- The page must work on a 375px-wide phone without horizontal scrolling.

## Content

- Wrap each part of the page in its own `<section>` element.
- Give the hero a headline, a subhead, and a call to action.

## Brand

- Use the colors and the font from the brand guide.
- Write the copy in the voice the brand guide describes.

## Images

- Do not embed or link real images. Use placeholder slots: `<img src="/placeholders/hero.png" data-image-brief="...">` for the hero, and `/placeholders/feature-1.png`, `/placeholders/feature-2.png`, `/placeholders/feature-3.png` for feature images.
- In each `data-image-brief` attribute, describe what the image should show. A separate tool fills the slots in later.
