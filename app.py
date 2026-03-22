import html
import gradio as gr
from rag import load_pdf, chunk_documents, build_vectorstore, get_retriever, get_llm, answer_question

# These will be reused across questions once a PDF is uploaded
retriever = None
llm = get_llm()

# ── 1. PDF UPLOAD HANDLER ─────────────────────────────────────
def process_pdf(file):
    """
    When user uploads a PDF:
    1. Load it
    2. Chunk it
    3. Build vector store
    4. Store retriever globally so we can use it for questions
    """
    global retriever

    documents = load_pdf(file.name)
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore)

    return f"✅ PDF processed! {len(chunks)} chunks created. Ready for questions."

# ── 2. QUESTION HANDLER ───────────────────────────────────────
def ask(question):
    """
    When user asks a question:
    1. Check if PDF was uploaded first
    2. Get answer from RAG pipeline
    3. Return answer + sources
    """
    if retriever is None:
        return "⚠️ Please upload a PDF first.", ""

    try:
        answer, sources = answer_question(question, retriever, llm)
    except Exception as e:
        return str(e), ""
    sources_text = "📄 Sources: " + ", ".join(sources)

    return answer, sources_text


# ── UI helpers (presentation only; call the handlers above) ────
def _sources_to_html(sources_text: str) -> str:
    hidden = '<div class="sources-wrap sources-wrap--hidden" aria-hidden="true"></div>'
    if not sources_text or not str(sources_text).strip():
        return hidden
    raw = str(sources_text)
    if "Sources:" in raw:
        part = raw.split("Sources:", 1)[1].strip()
    else:
        part = raw
    pages = [p.strip() for p in part.split(",") if p.strip()]
    if not pages:
        return hidden
    pills = "".join(f'<span class="source-pill">{html.escape(p)}</span>' for p in pages)
    return (
        '<div class="sources-wrap">'
        '<span class="sources-heading">Sources</span>'
        f'<div class="sources-row">{pills}</div>'
        "</div>"
    )


def process_pdf_ui(file):
    if file is None:
        return (
            '<div class="status-line">'
            '<span class="status-dot dot-muted"></span>'
            '<span class="status-text">Select a PDF, then run the pipeline.</span>'
            "</div>"
        )
    msg = process_pdf(file)
    dot = "dot-green" if ("✅" in msg or "ready" in msg.lower()) else "dot-amber"
    safe = html.escape(msg)
    return (
        f'<div class="status-line">'
        f'<span class="status-dot {dot}"></span>'
        f'<span class="status-text">{safe}</span>'
        f"</div>"
    )


def ask_ui(question):
    answer, sources_text = ask(question)
    return answer, _sources_to_html(sources_text)


def clear_state():
    """Reset RAG state when the uploaded PDF is removed from the file component."""
    global retriever
    retriever = None
    return (
        '<div class="status-line">'
        '<span class="status-dot dot-muted"></span>'
        '<span class="status-text">Select a PDF, then run the pipeline.</span>'
        "</div>"
    )


# Premium dark UI (Gradio 6 applies custom CSS via launch(css=...); SAAS_CSS is the single stylesheet.)
SAAS_CSS = """@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
* {
  font-family: 'DM Sans', sans-serif !important;
}
#rag-app h1, #rag-app h2, #rag-app h3, #rag-app h4, #rag-app h5, #rag-app h6,
h1, h2, h3, h4, h5, h6,
.gr-title, .title, .label-wrap label, .gr-form label span,
.syne-h1, .syne-h2, .panel-head, .how-it-works-btn,
.pb-step-title, .pb-modal-title, .sources-heading {
  font-family: 'Syne', sans-serif !important;
}
:root {
  --bg: #0a0a0f;
  --surface: rgba(18, 18, 26, 0.85);
  --border: rgba(255, 255, 255, 0.08);
  --text: #e8e8ef;
  --muted: #9898a8;
  --grad-a: #7c3aed;
  --grad-b: #f97316;
  --cyan-glow: rgba(34, 211, 238, 0.12);
}

/* Base canvas + orbs */
body, .dark body {
  background: var(--bg) !important;
}
#rag-app, #rag-app .gradio-container,
.gradio-container {
  color: var(--text) !important;
  background: transparent !important;
  width: 100% !important;
  max-width: none !important;
  margin: 0 auto !important;
  padding: 2rem 1.75rem 1.5rem !important;
  position: relative !important;
  z-index: 1 !important;
}
#rag-app label, #rag-app li,
.pb-step-body, .status-text, .panel textarea, .panel input,
button:not(.how-it-works-btn), .source-pill {
  font-family: 'DM Sans', sans-serif !important;
}
.pb-hero-wrap {
  margin-bottom: 2rem !important;
  padding-top: 0.25rem;
}

.gradio-container::before,
.gradio-container::after {
  content: "";
  position: fixed;
  pointer-events: none;
  z-index: 0;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.45;
}
.gradio-container::before {
  width: 420px;
  height: 420px;
  top: -120px;
  left: -80px;
  background: radial-gradient(circle, rgba(124, 58, 237, 0.55) 0%, transparent 70%);
}
.gradio-container::after {
  width: 380px;
  height: 380px;
  bottom: -100px;
  right: -60px;
  background: radial-gradient(circle, rgba(249, 115, 22, 0.45) 0%, transparent 70%);
}

footer { display: none !important; }

/* Third orb via wrapper */
.orb-cyan-wrap {
  position: fixed;
  width: 320px;
  height: 320px;
  top: 40%;
  left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 0;
  background: radial-gradient(circle, rgba(34, 211, 238, 0.18) 0%, transparent 65%);
  filter: blur(72px);
  opacity: 0.5;
}

/* Headings */
.syne-h1, .syne-h2 {
  letter-spacing: -0.02em;
}
.syne-h1 {
  font-size: 1.65rem !important;
  font-weight: 800 !important;
  margin: 0 0 0.35rem 0 !important;
  background: linear-gradient(135deg, #faf5ff 0%, #c4b5fd 40%, #fdba74 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.syne-h2 {
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  margin: 0 0 1rem 0 !important;
  line-height: 1.4 !important;
}
.pb-hero-wrap .syne-h2 {
  margin-bottom: 0.45rem !important;
}

/* Shell layout — left column height follows content (no dead space below status) */
.saas-shell {
  gap: 1.75rem !important;
  align-items: flex-start !important;
  margin-top: 0.25rem !important;
}
.panel {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.35rem 1.5rem !important;
  backdrop-filter: blur(12px);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset, 0 24px 48px rgba(0,0,0,0.35);
}
.panel-ingest.panel {
  align-self: flex-start !important;
  height: auto !important;
  min-height: 0 !important;
  background: rgba(14, 14, 20, 0.55) !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset !important;
  padding: 1.25rem 1.35rem !important;
}
.panel-ingest .panel-head {
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  margin-bottom: 1rem !important;
  opacity: 0.9;
}
.panel-ingest .file-holder,
.panel-ingest [data-testid="file-upload"] {
  margin-top: 0.25rem !important;
}
.panel-ingest .status-line {
  margin-top: 0.65rem !important;
  margin-bottom: 0 !important;
  background: rgba(10, 10, 15, 0.35) !important;
  border-color: rgba(255,255,255,0.06) !important;
  min-height: unset !important;
}
#rag-app .panel-ingest .html-container,
#rag-app .panel-ingest [class*="html"] {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
#rag-app .panel-head,
.panel-head {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted) !important;
  margin-bottom: 0.85rem !important;
}

/* Buttons */
button.btn-gradient {
  border: none !important;
  background: linear-gradient(90deg, var(--grad-a), var(--grad-b)) !important;
  color: #fff !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 0.55rem 1rem !important;
  box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
button.btn-gradient:hover {
  filter: brightness(1.06);
  box-shadow: 0 10px 28px rgba(249, 115, 22, 0.22);
  transform: translateY(-1px);
}

/* Inputs */
.panel textarea, .panel input[type="text"] {
  background: rgba(10, 10, 15, 0.6) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.92rem !important;
}
.panel .wrap label span {
  color: var(--muted) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
}

/* File upload zone */
.panel .file-preview, .panel [data-testid="file-upload"] {
  border-radius: 12px !important;
  border-color: var(--border) !important;
  background: rgba(10, 10, 15, 0.45) !important;
}

/* Status */
.status-line {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin-top: 0.5rem;
  padding: 0.65rem 0.75rem;
  border-radius: 10px;
  background: rgba(10, 10, 15, 0.55);
  border: 1px solid var(--border);
  min-height: 2.5rem;
}
.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-top: 0.35rem;
  flex-shrink: 0;
  box-shadow: 0 0 12px currentColor;
}
.dot-muted { background: #52525b; color: #52525b; }
.dot-amber { background: #f59e0b; color: #f59e0b; }
.dot-green { background: #22c55e; color: #22c55e; }
.status-text {
  font-size: 0.88rem;
  line-height: 1.45;
  color: #d4d4d8;
}

/* Chat answer */
.chat-answer textarea, .chat-answer input {
  min-height: 180px !important;
  line-height: 1.55 !important;
}

/* Sources HTML block */
.sources-wrap--hidden {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}
.sources-wrap {
  margin-top: 0.5rem;
}
.sources-heading {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  display: block;
  margin-bottom: 0.45rem;
}
.sources-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.source-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  font-family: 'DM Sans', sans-serif !important;
  background: linear-gradient(135deg, rgba(249, 115, 22, 0.25), rgba(249, 115, 22, 0.08));
  border: 1px solid rgba(249, 115, 22, 0.45);
  color: #fdba74;
}

/* Footer */
.saas-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 1.1rem 0 0.5rem;
  margin-top: 1.25rem;
  border-top: 1px solid var(--border);
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem;
  color: var(--muted);
}
.saas-footer .by {
  color: #a1a1b0;
}
.saas-footer a.icon-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  color: #c4b5fd;
  transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
}
.saas-footer a.icon-link:hover {
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(249, 115, 22, 0.2));
  color: #fff;
  transform: translateY(-1px);
}
.saas-footer svg {
  width: 18px;
  height: 18px;
  fill: currentColor;
}

/* Hide default Gradio branding row if any */
#rag-app .contain { padding-top: 0 !important; }

/* How it works — gradient text link */
.how-it-works-btn {
  display: inline-block;
  margin: 0.35rem 0 0.75rem 0;
  padding: 0;
  border: none;
  background: none;
  cursor: pointer;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.92rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  background-image: linear-gradient(90deg, var(--grad-a), var(--grad-b));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
  transition: filter 0.2s ease, transform 0.15s ease;
}
.how-it-works-btn:hover {
  filter: brightness(1.15);
  transform: translateX(2px);
}
.how-it-works-btn:focus-visible {
  outline: 2px solid rgba(124, 58, 237, 0.6);
  outline-offset: 3px;
}

/* Centered modal (How it works) */
.pb-modal-root {
  position: fixed;
  inset: 0;
  z-index: 99999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.32s ease;
}
.pb-modal-root.is-open {
  pointer-events: auto;
  opacity: 1;
}
.pb-modal-backdrop {
  position: absolute;
  inset: 0;
  z-index: 1;
  background: rgba(0, 0, 0, 0.7);
  opacity: 0;
  transition: opacity 0.32s ease;
}
.pb-modal-root.is-open .pb-modal-backdrop {
  opacity: 1;
}
.pb-modal-dialog {
  position: relative;
  z-index: 2;
  width: 100%;
  max-width: 600px;
  max-height: 80vh;
  background: #111118;
  border-radius: 16px;
  border-left: 3px solid rgba(124, 58, 237, 0.65);
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(255, 255, 255, 0.06) inset;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-sizing: border-box;
  opacity: 0;
  transform: scale(0.9);
  transition: opacity 0.32s ease, transform 0.32s cubic-bezier(0.34, 1.15, 0.64, 1);
}
.pb-modal-root.is-open .pb-modal-dialog {
  opacity: 1;
  transform: scale(1);
}
.pb-modal-close {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  z-index: 3;
  width: 40px;
  height: 40px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  background: rgba(22, 22, 32, 0.95);
  color: #e8e8ef;
  font-size: 1.35rem;
  line-height: 1;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  font-family: 'DM Sans', sans-serif !important;
}
.pb-modal-close:hover {
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(249, 115, 22, 0.2));
  border-color: rgba(124, 58, 237, 0.45);
  color: #fff;
}
.pb-modal-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 2.75rem 1.5rem 1.5rem 1.5rem;
  -webkit-overflow-scrolling: touch;
}
.pb-modal-title {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.2rem;
  font-weight: 800;
  text-align: left;
  margin: 0 0 1.35rem 0;
  padding-right: 2.5rem;
  background: linear-gradient(135deg, #faf5ff 0%, #c4b5fd 45%, #fdba74 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* How it works — step list (cards + gradient arrows) */
.pb-flow-vertical {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.pb-step-card {
  width: 100%;
  padding: 1rem 1rem 1rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(12, 12, 18, 0.85);
  box-sizing: border-box;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.pb-step-card:hover {
  border-color: rgba(124, 58, 237, 0.4);
  box-shadow: 0 0 16px rgba(124, 58, 237, 0.12);
}
.pb-step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.75rem;
  height: 1.75rem;
  margin-bottom: 0.65rem;
  border-radius: 999px;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.8rem;
  font-weight: 800;
  color: #fff;
  background: linear-gradient(135deg, var(--grad-a), var(--grad-b));
  box-shadow: 0 2px 10px rgba(124, 58, 237, 0.35);
}
.pb-step-title {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.92rem;
  font-weight: 700;
  color: #f4f4f8;
  margin: 0 0 0.5rem 0;
  line-height: 1.3;
}
.pb-step-body {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem;
  line-height: 1.55;
  color: #b4b4c4;
  margin: 0;
}
.pb-step-arrow {
  text-align: center;
  font-size: 24px;
  line-height: 1;
  margin: 0.5rem 0;
  padding: 0;
  background: linear-gradient(180deg, #7c3aed, #f97316);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
  font-family: 'Syne', sans-serif !important;
  user-select: none;
}

.pb-rag-footer {
  margin-top: 1.75rem;
  padding-top: 1.25rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  text-align: center;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem;
  font-style: italic;
  line-height: 1.55;
  background-image: linear-gradient(90deg, #c4b5fd, #fdba74, #fb923c);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}
"""

HERO_MODAL_HTML = """
<div class="pb-hero-wrap">
  <p class="syne-h1">PaperBrain</p>
  <p class="syne-h2">Upload a document. Ask anything.<br>Answers grounded in your file — not AI memory.</p>
  <button type="button" class="how-it-works-btn" id="pb-open-how" onclick="document.getElementById('paperbrain-modal-root').classList.add('is-open'); document.body.style.overflow='hidden';">How it works →</button>
</div>

<div id="paperbrain-modal-root" class="pb-modal-root">
  <div class="pb-modal-backdrop" onclick="document.getElementById('paperbrain-modal-root').classList.remove('is-open'); document.body.style.overflow='';"></div>
  <div class="pb-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="pb-modal-title">
    <button type="button" class="pb-modal-close" onclick="document.getElementById('paperbrain-modal-root').classList.remove('is-open'); document.body.style.overflow='';" aria-label="Close">×</button>
    <div class="pb-modal-scroll">
      <h2 id="pb-modal-title" class="pb-modal-title">How PaperBrain works</h2>

      <div class="pb-flow-vertical">
        <article class="pb-step-card">
          <span class="pb-step-num">1</span>
          <h3 class="pb-step-title">Your PDF is uploaded</h3>
          <p class="pb-step-body">PyMuPDF opens your file and reads every page as raw text. Think of it like copy-pasting the entire document into memory.</p>
        </article>
        <div class="pb-step-arrow" aria-hidden="true">↓</div>

        <article class="pb-step-card">
          <span class="pb-step-num">2</span>
          <h3 class="pb-step-title">Split into chunks</h3>
          <p class="pb-step-body">Your document is cut into pieces called chunks. Each chunk is 500 tokens long. A token is a small piece of text — not exactly a word, more like a syllable group. 100 tokens ≈ 75 words, so each chunk is roughly half a page. We overlap each chunk by 100 tokens with the next one so no sentence gets cut off at the boundary. Think of it like slicing a pizza where each slice slightly overlaps the next so no topping falls in the gap.</p>
        </article>
        <div class="pb-step-arrow" aria-hidden="true">↓</div>

        <article class="pb-step-card">
          <span class="pb-step-num">3</span>
          <h3 class="pb-step-title">Text converted to numbers</h3>
          <p class="pb-step-body">Each chunk is converted into a list of numbers called an embedding. These numbers capture the meaning of the text. Similar meaning = similar numbers. This is done locally using a free model called all-MiniLM-L6-v2. No API cost, runs on your machine.</p>
        </article>
        <div class="pb-step-arrow" aria-hidden="true">↓</div>

        <article class="pb-step-card">
          <span class="pb-step-num">4</span>
          <h3 class="pb-step-title">Stored in ChromaDB</h3>
          <p class="pb-step-body">All those numbers are saved in ChromaDB, a special database that can search by meaning not just by exact words. This is what makes RAG smart — you can ask a question in completely different words and it still finds the right chunks.</p>
        </article>
        <div class="pb-step-arrow" aria-hidden="true">↓</div>

        <article class="pb-step-card">
          <span class="pb-step-num">5</span>
          <h3 class="pb-step-title">Question becomes numbers too</h3>
          <p class="pb-step-body">When you type a question, it gets converted into numbers using the same embedding model. ChromaDB then finds the top 5 chunks whose numbers are closest to your question's numbers. That's semantic search.</p>
        </article>
        <div class="pb-step-arrow" aria-hidden="true">↓</div>

        <article class="pb-step-card">
          <span class="pb-step-num">6</span>
          <h3 class="pb-step-title">Groq builds the answer</h3>
          <p class="pb-step-body">The top 5 chunks are sent to Groq's Llama 3.3 70b model with one strict rule: answer using ONLY this context. If the answer isn't in the chunks, it says so. No hallucination. No guessing. Just your document.</p>
        </article>
      </div>

      <p class="pb-rag-footer">This is RAG — Retrieval Augmented Generation. Your AI, grounded in your documents.</p>
    </div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="saas-footer">
  <span class="by">Built by Quratulain Nayeem</span>
  <a class="icon-link" href="https://github.com/quratulain-nayeem" target="_blank" rel="noopener noreferrer" title="GitHub" aria-label="GitHub">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
  </a>
  <a class="icon-link" href="https://www.linkedin.com/in/quratulain-nayeem/" target="_blank" rel="noopener noreferrer" title="LinkedIn" aria-label="LinkedIn">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
  </a>
</div>
"""

# Escape closes the "How it works" modal (inline open/close works without this)
PAPERBRAIN_MODAL_JS = """
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') {
    var el = document.getElementById('paperbrain-modal-root');
    if (el && el.classList.contains('is-open')) {
      el.classList.remove('is-open');
      document.body.style.overflow = '';
    }
  }
});
"""

# ── 3. GRADIO UI ──────────────────────────────────────────────
with gr.Blocks(title="PaperBrain", elem_id="rag-app") as app:
    gr.HTML('<div class="orb-cyan-wrap" aria-hidden="true"></div>')
    gr.HTML(HERO_MODAL_HTML)

    with gr.Row(elem_classes=["saas-shell"]):
        with gr.Column(scale=5, elem_classes=["panel", "panel-ingest"]):
            gr.Markdown('<p class="panel-head">Ingest</p>')
            pdf_input = gr.File(label="PDF", file_types=[".pdf"], show_label=False)
            upload_btn = gr.Button("Run pipeline", elem_classes=["btn-gradient"])
            pipeline_status = gr.HTML(
                '<div class="status-line">'
                '<span class="status-dot dot-muted"></span>'
                '<span class="status-text">Select a PDF, then run the pipeline.</span>'
                "</div>"
            )

        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.Markdown('<p class="panel-head">Ask</p>')
            question_input = gr.Textbox(
                label="Question",
                placeholder="What does this document say about…",
                lines=2,
            )
            ask_btn = gr.Button("Get answer", elem_classes=["btn-gradient"])
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=12,
                elem_classes=["chat-answer"],
            )
            sources_output = gr.HTML(
                '<div class="sources-wrap sources-wrap--hidden" aria-hidden="true"></div>'
            )

    gr.HTML(FOOTER_HTML)

    upload_btn.click(process_pdf_ui, inputs=pdf_input, outputs=pipeline_status)
    pdf_input.clear(clear_state, outputs=pipeline_status)
    ask_btn.click(ask_ui, inputs=question_input, outputs=[answer_output, sources_output])

if __name__ == "__main__":
    app.launch(css=SAAS_CSS, js=PAPERBRAIN_MODAL_JS)
