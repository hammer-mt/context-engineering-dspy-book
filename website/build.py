#!/usr/bin/env python3
"""Build the static book site into website/public/.

    pip install markdown pygments
    python3 website/build.py

Deploy website/public/ to Cloudflare Pages (no build step needed there).
"""
import html
import json
import re
from pathlib import Path

import markdown

# ---- settings -------------------------------------------------------------
SITE = "https://context-engineering-dspy.mike-190.workers.dev"  # change when a custom domain is attached
OREILLY = "https://www.oreilly.com/library/view/context-engineering-with/0642572261603/"
AMAZON = "https://a.co/d/04HgIVje"
CHAPTER_SLUG = "dspy-in-eight-steps"  # unlisted: not linked, not in the sitemap, noindex
CHAPTER_INDEXABLE = False  # set True to let search engines index the free chapter
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
CHAPTER_MD = HERE / "content" / "dspy-in-eight-steps.md"

FAQ = [
    ("What is DSPy?",
     "DSPy stands for Declarative Self-improving Python. It is an open source framework, released under the MIT license in October 2023 and developed from work by Omar Khattab and colleagues at Stanford. Instead of writing a prompt, you define an AI task by its inputs and outputs in a Signature, combine tasks into modules, and use optimizers to improve the program against a dataset and an evaluation metric."),
    ("What is context engineering?",
     "Context engineering is the discipline of systematically curating what information a language model receives, to maximize performance while minimizing costs and errors. The context is everything the model can see when it responds: system instructions, messages, conversation history, tool definitions and results, retrieved documents, memories and examples."),
    ("What is the difference between context engineering and prompt engineering?",
     "Prompt engineering is concerned with changes inside the prompt: instructions, examples and fixed chains of prompts. Context engineering also covers the wider system around the prompt: retrieving documents and memories, formatting content, designing the harness, and pruning irrelevant information on long-running tasks. The book treats prompt engineering as a subset of context engineering."),
    ("What is context rot?",
     "Context rot is the way model performance degrades as the input gets longer, a term from the vector database company Chroma. Chapter 1 describes 11 ways long contexts fail, including context distraction, lost in the middle, context confusion, context clash, context drift and context poisoning, and 10 techniques for fixing them."),
    ("What is GEPA in DSPy?",
     "GEPA (Genetic-Pareto) is a DSPy optimizer that rewrites your prompt instructions by reflecting on where the program failed. It is the author’s default optimizer once a dataset has 10 or more examples. In the book’s benchmark of optimization approaches on one AI-detection task, GEPA took accuracy from 53.75% to 80% for about $0.58."),
    ("What is harness engineering, and how does it relate to context engineering?",
     "Harness engineering is OpenAI’s term for the runtime scaffolding around a model: the environment, tools, feedback loops and control systems that deliver context to an agent and act on its outputs. The book’s view is that this is still context engineering. The harness is what delivers the context, and getting the context right is what delivers performance."),
    ("What is prompt optimization?",
     "Prompt optimization means improving a prompt against a dataset and an evaluation metric instead of by trial and error. In DSPy, a prompt optimizer selects or generates few-shot examples, proposes and tests better instructions, or both. The book’s Chapter 1 example cuts a 1,784-token prompt to 43 tokens while accuracy moves from 64% to 67%."),
    ("What is LLM-as-a-judge?",
     "LLM-as-a-judge uses one language model to evaluate another model’s output, which makes it possible to score subjective tasks like writing. The book’s method is to label 20 to 50 outputs yourself, train the judge on those labels until it agrees with you 80 to 90 percent of the time, and then use the judge as the metric an optimizer works against."),
    ("What is MIPROv2?",
     "MIPROv2 is a DSPy optimizer that searches over both instructions and few-shot examples. The book recommends trying it when you have a large dataset of 200 or more examples and a substantial optimization budget. In the Chapter 6 benchmark it reached 80% on validation but 66.25% on the locked test, a reminder to hold out data."),
    ("How does DSPy compare with LangChain or Pydantic AI?",
     "Chapter 8 builds the same agent in DSPy, LangChain, Pydantic AI, the OpenAI Agents SDK and CrewAI. The book’s argument for DSPy is that every agent is a DSPy program, so it can be evaluated and optimized. You don’t have to choose one: many teams optimize a prompt with DSPy, export it, and run it in the framework they already use, which Chapter 10 covers."),
    ("Do I need to know DSPy before reading the book?",
     "No. Chapter 1 covers context engineering without any DSPy, Chapter 2 introduces the framework’s five core abstractions, and Chapter 3 builds a first program end to end. The examples are in Python, and the concepts transfer to DSPy ports such as Ax for TypeScript, dspy.rb for Ruby and dspy-go."),
    ("How many examples do I need to optimize a prompt with DSPy?",
     "Fewer than most people expect. The book’s guidance is that 10 to 50 well-chosen examples combined with an optimizer will often outperform a handcrafted prompt, and that you can start with 10 to 20 and grow the dataset from the errors you see in production."),
    ("Is DSPy used in production?",
     "Yes. The book cites Shopify using DSPy with GEPA to extract structured metadata across every shop, Dropbox optimizing the relevance judge behind Dash, and Databricks building DSPy-powered judges and RAG pipelines, along with JetBlue, Replit and Moody’s."),
    ("What are recursive language models (RLMs)?",
     "A recursive language model puts a large input into a code environment and calls the model on smaller parts of it, rather than loading everything into one context window. DSPy provides the pattern as dspy.RLM. The book uses it for documents too large to fit in context, such as financial filings."),
    ("Can I optimize Claude skills, SKILL.md, CLAUDE.md or AGENTS.md files with DSPy?",
     "Yes, and Chapter 11 is about exactly that. A skill is a Markdown instruction file (SKILL.md) that a coding agent such as Claude Code, Codex or Cursor loads for a task, so it can be tested against failed sessions and rewritten by GEPA. The same approach works for the always-loaded AGENTS.md or CLAUDE.md file and for agent persona files."),
    ("Which version of DSPy does the book use?",
     "The companion repository pins DSPy 3.3.0 and supports Python 3.12 through 3.14. Each chapter has notebooks you can run locally, and many run in Google Colab."),
]


def faq_html():
    return "\n  ".join(
        f"<details><summary>{html.escape(q)}</summary><div><p>{html.escape(a)}</p></div></details>" for q, a in FAQ)


def faq_json():
    return ",".join(json.dumps({"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}}, ensure_ascii=False) for q, a in FAQ)


def fill(text):
    return (text.replace("__FAQ_HTML__", faq_html()).replace("__FAQ_JSON__", faq_json())
            .replace("__SITE__", SITE).replace("__OREILLY__", OREILLY).replace("__AMAZON__", AMAZON))


FIGURE_SIZES = {"1": (1600, 657), "2": (1600, 797), "3": (1600, 1261)}


def chapter_html():
    src = CHAPTER_MD.read_text()
    src = src.split("\n---\n", 1)[1] if src.startswith("---") else src
    src = re.sub(r"\A\s*# .+\n", "", src)

    # program output and prompt dumps are printed plain in the book, so don't colour them as Python
    def plain_output(m):
        block = m.group(2)
        is_output = "[[ ## " in block or re.search(r"^\d{4}/\d\d/\d\d ", block, flags=re.M) or "Average Metric" in block
        return f"```{'text' if is_output else m.group(1)}\n{block}```"
    src = re.sub(r"```(python|json)\n(.*?)```", plain_output, src, flags=re.S)
    body = markdown.markdown(src, extensions=["fenced_code", "codehilite", "tables", "sane_lists"],
                             extension_configs={"codehilite": {"guess_lang": False, "css_class": "code"}})
    body = body.replace("<table>", '<div class="tablebox"><table>').replace("</table>", "</table></div>")
    # figures: the Markdown keeps only captions, so put each image back above its caption
    def figure(m):
        n = m.group(1)
        cap = re.sub(r"</?em>", "", m.group(2)).strip()
        alt = html.escape(re.sub(r"<[^>]+>", "", cap), quote=True)
        w, h = FIGURE_SIZES[n]
        return (f'<figure id="figure-3-{n}"><img src="figure-3-{n}.png" alt="{alt}" width="{w}" height="{h}">'
                f'<figcaption>Figure 3-{n}. {cap}</figcaption></figure>')
    body = re.sub(r"(?<!<em>)Figure 3-(\d+)", r'<a href="#figure-3-\1">Figure 3-\1</a>', body)
    body = re.sub(r"<p><em>Figure 3-(\d+)\.(.*?)</p>", figure, body, flags=re.S)
    # callouts: the book sets these beside an animal icon
    body = body.replace("<blockquote>", '<aside class="note"><img src="note.png" alt="" width="62" height="81"><div>').replace("</blockquote>", "</div></aside>")
    return body


def build_chapter():
    robots = "" if CHAPTER_INDEXABLE else '<meta name="robots" content="noindex, nofollow">\n'
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSPy in Eight Steps | Chapter 3 of Context Engineering with DSPy</title>
<meta name="description" content="Chapter 3 of Context Engineering with DSPy by Mike Taylor: build and optimize your first DSPy program end to end with Signatures, modules, an LLM judge and the GEPA optimizer.">
{robots}<link rel="canonical" href="{SITE}/{CHAPTER_SLUG}/">
<meta property="og:title" content="DSPy in Eight Steps">
<meta property="og:description" content="A free chapter from Context Engineering with DSPy (O’Reilly) by Mike Taylor.">
<meta property="og:image" content="{SITE}/cover.jpg">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,700;1,400&family=Open+Sans:wdth,wght@75,600;75,700&family=Ubuntu+Mono:ital,wght@0,400;0,700;1,400&display=swap">
<link rel="stylesheet" href="/book.css">
</head>
<body>
<div class="strip"><div>
  <a href="/"><img src="/cover.jpg" width="38" height="50" alt=""></a>
  <p>A free chapter from <a href="/"><em>Context Engineering with DSPy</em></a> by Mike Taylor</p>
  <p class="links"><a href="/{CHAPTER_SLUG}.pdf">Download PDF</a><a href="{OREILLY}">O’Reilly</a><a href="{AMAZON}">Amazon</a></p>
</div></div>
<main class="page">
  <header class="opener"><p>Chapter 3</p><h1>DSPy in Eight Steps</h1></header>
  <article>
{chapter_html()}
  </article>
  <footer class="running"><span>Chapter 3: DSPy in Eight Steps</span><span>Context Engineering with DSPy</span></footer>
</main>
<div class="strip end"><div>
  <a href="/"><img src="/cover.jpg" width="38" height="50" alt=""></a>
  <p>That was Chapter 3 of 11. Chapter 4 picks up where this one ends: where to find the examples for your dataset. <a href="https://www.youtube.com/live/k1mmJaqQky0">Watch Mike work through this chapter live</a>, or run the notebook in the <a href="https://github.com/hammer-mt/context-engineering-dspy-book">companion repository</a>.</p>
  <p class="links"><a href="{OREILLY}">Read on O’Reilly</a><a href="{AMAZON}">Buy on Amazon</a></p>
</div></div>
<p class="legal">© Saxifrage, LLC. Excerpted from <em>Context Engineering with DSPy</em>, published by O’Reilly Media. Shared by the author.</p>
</body>
</html>
"""
    out = PUBLIC / CHAPTER_SLUG
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)


def main():
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC / "index.html").write_text(fill((HERE / "src" / "index.html").read_text()))
    build_chapter()
    urls = [f"{SITE}/"] + ([f"{SITE}/{CHAPTER_SLUG}/"] if CHAPTER_INDEXABLE else [])
    (PUBLIC / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                        + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    (PUBLIC / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n")
    headers = "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n\n/cover.jpg\n  Cache-Control: public, max-age=604800\n"
    if not CHAPTER_INDEXABLE:
        headers += f"\n/{CHAPTER_SLUG}/*\n  X-Robots-Tag: noindex\n\n/{CHAPTER_SLUG}.pdf\n  X-Robots-Tag: noindex\n"
    (PUBLIC / "_headers").write_text(headers)
    (PUBLIC / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="5" fill="#0b7a80"/><text x="16" y="22" font-family="Arial,sans-serif" font-size="15" font-weight="700" text-anchor="middle" fill="#fff">CE</text></svg>\n')
    (PUBLIC / "404.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Page not found | Context Engineering with DSPy</title><link rel="stylesheet" href="/styles.css"></head><body><main class="chapter"><div class="wrap"><h1>Page not found</h1><p><a href="/">Back to the book</a></p></div></main></body></html>\n')
    print("built", sorted(p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
