// app.js - talks to the FastAPI endpoints in src/app.py. no scoring logic here,
// the server calls the same functions evaluate.py uses.

let QUESTIONS = [];
let ACTIVE_TICKER = null;
let LAST = null;   // last /api/ask result, so live scoring has something to score


function showTab(which) {
  document.getElementById("view-ask").classList.toggle("hide", which !== "ask");
  document.getElementById("view-eval").classList.toggle("hide", which !== "eval");
  document.getElementById("tab-ask").classList.toggle("on", which === "ask");
  document.getElementById("tab-eval").classList.toggle("on", which === "eval");
}


function fmt(v) {
  if (v === null || v === undefined) {
    return "-";
  }
  return Number(v).toFixed(2);
}


function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
  });
}


async function boot() {
  const cfg = await (await fetch("/api/config")).json();
  document.getElementById("framing").textContent =
    "Both systems use " + cfg.chat_model + " at temperature 0. The only difference is whether the top-"
    + cfg.top_k + " retrieved filing passages are supplied as context.";

  const companies = await (await fetch("/api/companies")).json();
  QUESTIONS = await (await fetch("/api/questions")).json();

  const chips = document.getElementById("chips");
  for (let i = 0; i < companies.length; i++) {
    const c = companies[i];
    const b = document.createElement("button");
    b.className = "chip";
    if (i === 0) {
      b.className = "chip on";
      ACTIVE_TICKER = c.ticker;
    }
    b.textContent = c.ticker;
    b.title = c.name;
    b.addEventListener("click", function () { setTicker(c.ticker, b); });
    chips.appendChild(b);
  }

  document.getElementById("tab-ask").addEventListener("click", function () { showTab("ask"); });
  document.getElementById("tab-eval").addEventListener("click", function () { showTab("eval"); });
  document.getElementById("qsel").addEventListener("change", pickQuestion);
  document.getElementById("askbtn").addEventListener("click", ask);
  document.getElementById("scorebtn").addEventListener("click", scoreLive);

  fillQuestions();
  loadEval();
}


// the chips only choose which questions to show. retrieval still searches all
// five filings - that's the point of the no-filter design decision.
function setTicker(ticker, btn) {
  ACTIVE_TICKER = ticker;
  const all = document.querySelectorAll(".chip");
  for (let i = 0; i < all.length; i++) {
    all[i].classList.remove("on");
  }
  btn.classList.add("on");
  fillQuestions();
  pickQuestion();
}


function fillQuestions() {
  const sel = document.getElementById("qsel");
  sel.innerHTML = "";

  const own = document.createElement("option");
  own.value = "";
  own.textContent = "(type my own question)";
  sel.appendChild(own);

  for (let i = 0; i < QUESTIONS.length; i++) {
    const q = QUESTIONS[i];
    if (q.company !== ACTIVE_TICKER) {
      continue;
    }
    const o = document.createElement("option");
    o.value = q.id;
    o.textContent = q.id + "  -  " + q.question;
    sel.appendChild(o);
  }
}


function currentPair() {
  const id = document.getElementById("qsel").value;
  if (!id) {
    return null;
  }
  for (let i = 0; i < QUESTIONS.length; i++) {
    if (QUESTIONS[i].id === id) {
      return QUESTIONS[i];
    }
  }
  return null;
}


function pickQuestion() {
  const pair = currentPair();
  const truth = document.getElementById("truth");

  // clear anything from the previous question
  document.getElementById("panels").classList.add("hide");
  document.getElementById("ctxwrap").classList.add("hide");
  document.getElementById("scorebar").classList.add("hide");

  if (!pair) {
    document.getElementById("qbox").value = "";
    truth.classList.add("hide");
    return;
  }

  document.getElementById("qbox").value = pair.question;
  truth.classList.remove("hide");
  truth.innerHTML = "<b>Ground truth (" + pair.id + ", " + pair.type + "):</b> "
    + esc(pair.ground_truth)
    + '<div class="src">Source: ' + esc(pair.source) + "</div>";
}


function metricRow(el, items, scope) {
  el.innerHTML = "";
  for (let i = 0; i < items.length; i++) {
    const d = document.createElement("div");
    d.className = "m";
    d.innerHTML = items[i][0] + " <b>" + fmt(items[i][1]) + "</b>";
    el.appendChild(d);
  }
  if (scope) {
    const s = document.createElement("div");
    s.className = "scope";
    s.textContent = scope;
    el.appendChild(s);
  }
  el.classList.remove("hide");
}


// scores from the frozen evaluation run - instant, and the same numbers the
// dissertation reports
function showSavedScores(pair) {
  const ragEl = document.getElementById("rag-metrics");
  const baseEl = document.getElementById("base-metrics");

  if (!pair || !pair.saved) {
    ragEl.classList.add("hide");
    baseEl.classList.add("hide");
    return;
  }

  const g = pair.saved.rag_ragas || {};
  const b = pair.saved.baseline_ragas || {};

  metricRow(ragEl, [
    ["Faith", g.faithfulness],
    ["AnsRel", g.answer_relevancy],
    ["CtxPrec", g.context_precision],
    ["CtxRec", g.context_recall],
    ["EM", pair.saved.rag_em],
    ["F1", pair.saved.rag_f1]
  ], "from the frozen evaluation run");

  metricRow(baseEl, [
    ["AnsRel", b.answer_relevancy],
    ["EM", pair.saved.baseline_em],
    ["F1", pair.saved.baseline_f1]
  ], "context metrics N/A - the baseline retrieves nothing");
}


async function ask() {
  const q = document.getElementById("qbox").value.trim();
  if (!q) {
    return;
  }

  const btn = document.getElementById("askbtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Asking...';

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    LAST = { question: q, data: data };

    document.getElementById("rag-answer").textContent = data.rag_answer;
    document.getElementById("base-answer").textContent = data.baseline_answer;
    document.getElementById("panels").classList.remove("hide");

    const pair = currentPair();
    showSavedScores(pair);
    renderChunks(data);

    document.getElementById("scorebar").classList.remove("hide");
    if (pair) {
      document.getElementById("scorenote").textContent =
        "Re-scores this question with gpt-4o. Takes a couple of minutes.";
    } else {
      document.getElementById("scorenote").textContent =
        "Custom question: only faithfulness and answer relevance can be computed.";
    }
  } catch (e) {
    document.getElementById("rag-answer").textContent = "Error: " + e;
    document.getElementById("panels").classList.remove("hide");
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask both systems";
  }
}


function renderChunks(data) {
  const wrap = document.getElementById("chunks");
  wrap.innerHTML = "";

  let own = 0;
  for (let i = 0; i < data.tickers.length; i++) {
    if (data.tickers[i] === ACTIVE_TICKER) {
      own++;
    }
  }

  document.getElementById("ctxsummary").textContent =
    own + " of " + data.tickers.length + " chunks came from " + ACTIVE_TICKER
    + ". Retrieval is not filtered by company, so passages from other filings can appear.";

  for (let i = 0; i < data.contexts.length; i++) {
    const t = data.tickers[i];
    const other = (t !== ACTIVE_TICKER);

    let tagClass = "tag";
    let tagText = esc(t);
    if (other) {
      tagClass = "tag other";
      tagText = esc(t) + " - different company";
    }

    const d = document.createElement("details");
    d.className = "chunk";
    d.innerHTML = "<summary>[" + (i + 1) + '] <span class="' + tagClass + '">'
      + tagText + "</span></summary><pre>" + esc(data.contexts[i]) + "</pre>";
    wrap.appendChild(d);
  }

  document.getElementById("ctxwrap").classList.remove("hide");
}


// prove the scores aren't hardcoded. slow, because faithfulness and context
// recall each make several judge calls.
async function scoreLive() {
  if (!LAST) {
    return;
  }

  const pair = currentPair();
  const btn = document.getElementById("scorebtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spin dark"></span>Scoring...';

  let truth = "";
  if (pair) {
    truth = pair.ground_truth;
  }

  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: LAST.question,
        rag_answer: LAST.data.rag_answer,
        baseline_answer: LAST.data.baseline_answer,
        contexts: LAST.data.contexts,
        ground_truth: truth
      })
    });
    const s = await res.json();
    const g = s.rag_ragas || {};
    const b = s.baseline_ragas || {};

    let scope = "scored live just now";
    if (s.note) {
      scope = scope + " - " + s.note;
    }

    metricRow(document.getElementById("rag-metrics"), [
      ["Faith", g.faithfulness],
      ["AnsRel", g.answer_relevancy],
      ["CtxPrec", g.context_precision],
      ["CtxRec", g.context_recall]
    ], scope);

    metricRow(document.getElementById("base-metrics"), [
      ["AnsRel", b.answer_relevancy]
    ], "scored live just now");
  } catch (e) {
    document.getElementById("scorenote").textContent = "Scoring failed: " + e;
  } finally {
    btn.disabled = false;
    btn.textContent = "Score this live with RAGAS";
  }
}


async function loadEval() {
  const body = document.getElementById("evalbody");
  const res = await (await fetch("/api/summary")).json();

  if (!res.available) {
    body.innerHTML = "<p>No frozen results found. Run "
      + "<code>python3 src/evaluate.py --run-name final</code> first.</p>";
    return;
  }

  const s = res.summary;
  const cov = s.overall.coverage || {};
  let html = "";

  html += "<h2>RAGAS coverage</h2>";
  html += '<div class="sub">How many questions each metric actually scored. '
    + "A mean over part of the set is not the same claim as a mean over all of it.</div>";
  html += "<table><tr><th>metric</th><th>scored</th></tr>";

  const order = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];
  for (let i = 0; i < order.length; i++) {
    const k = order[i];
    html += "<tr><td>" + k + "</td><td>" + (cov[k] || "-") + "</td></tr>";
  }
  html += "</table>";

  html += "<h2>Overall (n=" + s.overall.n + ")</h2>" + twoCol(s.overall);
  html += "<h2>By question type</h2>" + breakdown(s.by_type);
  html += "<h2>By company</h2>" + breakdown(s.by_company);

  body.innerHTML = html;
}


function twoCol(agg) {
  const r = agg.rag;
  const b = agg.baseline;

  const rows = [
    ["Faithfulness", r.faithfulness, null],
    ["Answer relevance", r.answer_relevancy, b.answer_relevancy],
    ["Context precision", r.context_precision, null],
    ["Context recall", r.context_recall, null],
    ["Exact Match", r.em, b.em],
    ["F1", r.f1, b.f1]
  ];

  let h = "<table><tr><th>metric</th><th>RAG</th><th>Baseline</th></tr>";
  for (let i = 0; i < rows.length; i++) {
    const x = rows[i];
    let baseCell = "-";
    if (x[2] !== null) {
      baseCell = fmt(x[2]);
    }
    h += "<tr><td>" + x[0] + "</td><td>" + fmt(x[1]) + "</td><td>" + baseCell + "</td></tr>";
  }
  h += "</table>";

  h += '<div class="sub" style="margin-top:8px">Own-company chunks: <b>'
    + r.own_company_chunks
    + "/5</b> - a diagnostic for the no-ticker-filter decision, not a score.</div>";
  return h;
}


function breakdown(group) {
  let h = "<table><tr><th></th><th>n</th><th>Faith</th><th>AnsRel</th><th>CtxPrec</th>"
    + "<th>CtxRec</th><th>RAG EM</th><th>RAG F1</th><th>Base EM</th><th>Base F1</th></tr>";

  const keys = Object.keys(group);
  for (let i = 0; i < keys.length; i++) {
    const a = group[keys[i]];
    const r = a.rag;
    const b = a.baseline;
    h += "<tr><td>" + keys[i] + "</td><td>" + a.n + "</td><td>" + fmt(r.faithfulness)
      + "</td><td>" + fmt(r.answer_relevancy) + "</td><td>" + fmt(r.context_precision)
      + "</td><td>" + fmt(r.context_recall) + "</td><td>" + fmt(r.em)
      + "</td><td>" + fmt(r.f1) + "</td><td>" + fmt(b.em)
      + "</td><td>" + fmt(b.f1) + "</td></tr>";
  }
  h += "</table>";
  return h;
}


boot();
