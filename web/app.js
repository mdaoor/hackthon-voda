/* Lifestyle Companion - vanilla JS client (no build step). */
const S = { user: null, cfg: null, pendingCheckout: null, busy: false, sttMode: "transcribe",
  voiceLanguage: "auto", lastDetectedLanguage: null, categoriesLoaded: false };
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cur = () => (S.cfg?.currency === "DEMO_UNITS" ? "" : " " + (S.cfg?.currency || ""));

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const body = isJson ? await res.json() : await res.blob();
  if (!res.ok) throw Object.assign(new Error(body?.detail || body?.error || res.statusText), { status: res.status, body });
  return body;
}
const post = (path, data) => api(path, { method: "POST", body: JSON.stringify(data) });

/* ------------------------------------------------------------------ markdown-lite */
function md(text) {
  const lines = esc(text).split(/\n/);
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)\*(\S.+?)\*/g, "$1<em>$2</em>");
    const li = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (li) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${li[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${line.replace(/^#+\s*/, "")}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

/* ------------------------------------------------------------------ entry */
async function boot() {
  S.cfg = await api("/api/config");
  S.sttMode = S.cfg.stt;
  const configured = new Set(S.cfg.configured_voice_languages || []);
  ["en-US", "ar-SA"].forEach((code) => {
    const option = $(`#voice-language option[value="${code}"]`);
    option.disabled = configured.size > 0 && !configured.has(code);
  });
  if (!S.cfg.voice_auto_detection) {
    $("#voice-language option[value=auto]").disabled = true;
    S.voiceLanguage = S.cfg.configured_voice_languages?.[0] || "en-US";
    $("#voice-language").value = S.voiceLanguage;
  }
  if (S.cfg.sample_ids?.length) {
    $("#samples").hidden = false;
    $("#sample-list").innerHTML = S.cfg.sample_ids.map((id) => `<button type="button" data-id="${esc(id)}">${esc(id)}</button>`).join("");
    $("#sample-list").onclick = (e) => { if (e.target.dataset.id) { $("#user-id").value = e.target.dataset.id; openCustomer(e.target.dataset.id); } };
  }
  const last = sessionStorage.getItem("companion_user");
  if (last) openCustomer(last, true);
}

$("#entry-form").addEventListener("submit", (e) => { e.preventDefault(); openCustomer($("#user-id").value.trim()); });

async function openCustomer(id, silent = false) {
  const err = $("#entry-error");
  err.hidden = true;
  if (!id) { err.textContent = "Enter a customer ID to continue."; err.hidden = false; return; }
  const btn = $("#entry-form .btn");
  btn.disabled = true; btn.textContent = "Opening…";
  try {
    const data = await post("/api/session/start", { user_id: id });
    S.user = data.customer.user_id;
    sessionStorage.setItem("companion_user", S.user);
    $("#entry").hidden = true; $("#app").hidden = false;
    renderHeader(data.customer);
    $("#messages").innerHTML = "";
    S.pendingCheckout = data.pending_checkout;
    data.transcript.forEach(renderEntry);
    renderBasket(data.basket);
    refreshPanel();
    scrollDown();
    $("#input").focus();
  } catch (e) {
    sessionStorage.removeItem("companion_user");
    if (!silent) { err.textContent = e.status === 404 ? e.message : `Couldn't open the companion: ${e.message}`; err.hidden = false; }
  } finally { btn.disabled = false; btn.textContent = "Open companion"; }
}

function renderHeader(c) {
  $("#who-id").textContent = c.user_id;
  $("#who-meta").textContent = [c.region, `${c.membership_tier} member`, `household of ${c.household_size}`, c.device].filter(Boolean).join(", ");
}

$("#btn-switch").onclick = () => {
  sessionStorage.removeItem("companion_user"); stopSpeaking();
  S.user = null; $("#app").hidden = true; $("#entry").hidden = false; $("#user-id").value = ""; $("#user-id").focus();
};
$("#btn-new").onclick = async () => {
  if (!S.user || S.busy) return;
  setBusy(true);
  try {
    const data = await post("/api/session/reset", { user_id: S.user });
    $("#messages").innerHTML = ""; S.pendingCheckout = null;
    data.transcript.forEach(renderEntry); renderBasket(data.basket); refreshPanel();
  } finally { setBusy(false); }
};

/* ------------------------------------------------------------------ chat */
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); sendText(); });
$("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); } });
$("#input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"; });

function sendText() {
  const t = $("#input").value.trim();
  if (!t || S.busy) return;
  $("#input").value = ""; $("#input").style.height = "auto";
  send(t, "text");
}

async function send(text, channel) {
  stopSpeaking();
  renderEntry({ role: "user", text, channel });
  setBusy(true);
  try {
    const voiceMeta = channel === "voice" ? {
      voice_language: S.lastDetectedLanguage || (S.voiceLanguage === "auto" ? null : S.voiceLanguage),
      voice_languages: S.lastDetectedLanguage ? [S.lastDetectedLanguage] : []
    } : {};
    const r = await post("/api/chat", { user_id: S.user, message: text, channel, ...voiceMeta });
    S.pendingCheckout = r.pending_checkout;
    renderEntry(r);
    renderBasket(r.basket);
    refreshPanel();
    if (channel === "voice" || $("#speak-replies").checked) speak(r.text, r.language);
  } catch (e) {
    renderEntry({ role: "assistant", text: `Something went wrong: ${e.message}. Please try again.`, error: true });
  } finally { setBusy(false); }
}

function setBusy(b) {
  S.busy = b;
  $("#typing").hidden = !b;
  $("#composer .send").disabled = b;
  if (b) scrollDown();
}
const scrollDown = () => requestAnimationFrame(() => { const m = $("#messages"); m.scrollTop = m.scrollHeight; });

function renderEntry(e) {
  const el = document.createElement("div");
  el.dir = "auto";
  if (e.role === "user") {
    el.className = "msg user";
    const via = e.channel === "voice" ? "Spoken" : e.channel === "ui" ? "Button" : "";
    el.innerHTML = `${esc(e.text)}${via ? `<span class="via">${via}</span>` : ""}`;
  } else {
    el.className = "msg assistant" + (e.error ? " error-msg" : "");
    let html = `<div class="text" dir="auto">${md(e.text || "")}</div>`;
    (e.cards || []).forEach((c) => { html += optionsHtml(c); });
    if (e.checkout) html += checkoutHtml(e.checkout);
    if (e.order) html += orderHtml(e.order);
    if (e.actions?.length) html += actionsHtml(e.actions);
    el.innerHTML = html;
  }
  $("#messages").appendChild(el);
  scrollDown();
}

function priceHtml(p) {
  return `<span class="price">${esc(p.final_price)}${cur()}${p.discount_pct ? `<s>${esc(p.list_price)}</s><span class="off">−${p.discount_pct}%</span>` : ""}</span>`;
}
function productHtml(p, n) {
  const flags = [
    p.eligible === false ? `<span class="badge no">Not available: ${esc((p.ineligible_reason || "").replace(/_/g, " ").toLowerCase())}</span>` : "",
    p.subscription ? `<span class="badge sub">Subscription, can't check out yet</span>` : "",
  ].join("");
  return `<article class="product">
    ${n ? `<span class="n">Option ${n}</span>` : ""}
    <span class="name">${esc(p.name)}</span>
    <span class="cat">${esc((p.category || "").replace(/_/g, " "))}${p.brand ? `, brand ${esc(p.brand)}` : ""}</span>
    ${priceHtml(p)}
    <span class="quality">Quality ${p.quality_tier} of 5</span>
    ${p.why?.length ? `<span class="why">${esc(p.why.slice(0, 2).join("; "))}</span>` : ""}
    ${flags}
    ${p.eligible !== false ? `<button class="btn small" data-add="${esc(p.product_id)}" data-name="${esc(p.name)}" data-n="${n || ""}">Add to basket</button>` : ""}
  </article>`;
}
function optionsHtml(c) {
  return `<div class="options"><h3>${esc(c.title)}</h3><div class="option-row">${c.products.map((p, i) => productHtml(p, i + 1)).join("")}</div></div>`;
}
function checkoutHtml(ch) {
  const s = ch.summary;
  const live = S.pendingCheckout && S.pendingCheckout.checkout_id === ch.checkout_id;
  const rows = s.items.map((i) => `<tr><td>${esc(i.product_name)} × ${i.quantity}${i.discount_rate !== "0" && i.discount_rate !== "0.0" ? ` <span class="off">−${Math.round(parseFloat(i.discount_rate) * 100)}%</span>` : ""}</td><td class="num">${esc(i.line_total)}${cur()}</td></tr>`).join("");
  return `<div class="summary" data-checkout="${esc(ch.checkout_id)}">
    <h3>Checkout summary</h3>
    <table>${rows}<tr class="total"><td>Total</td><td class="num">${esc(s.total)}${cur()}</td></tr></table>
    <p class="note">Simulated purchase. No payment is taken.</p>
    ${live ? `<div class="row-actions"><button class="btn primary" data-confirm="${esc(ch.checkout_id)}">Confirm order</button><button class="btn" data-cancel="1">Not yet</button></div>` : ""}
  </div>`;
}
function orderHtml(o) {
  return `<div class="summary done"><h3>Order placed</h3>
    <table>${o.items.map((i) => `<tr><td>${esc(i.product_name)} × ${i.quantity}</td><td class="num">${esc(i.line_total)}${cur()}</td></tr>`).join("")}
    <tr class="total"><td>Total</td><td class="num">${esc(o.total)}${cur()}</td></tr></table>
    <p class="note">Order ${esc(o.order_id)}. Simulated, no payment processed.</p></div>`;
}
function actionsHtml(actions) {
  const items = actions.map((a) => {
    const args = Object.keys(a.input || {}).length ? ` <code>${esc(JSON.stringify(a.input)).slice(0, 160)}</code>` : "";
    return `<li class="${a.ok ? "" : "fail"}"><strong>${esc(a.tool)}</strong>${args}: ${esc(a.summary)}</li>`;
  }).join("");
  return `<details class="actions"><summary>${actions.length} action${actions.length > 1 ? "s" : ""} taken</summary><ol>${items}</ol></details>`;
}

// Delegated clicks for cards + checkout buttons
document.addEventListener("click", async (e) => {
  const add = e.target.closest("[data-add]");
  if (add && !S.busy) {
    const n = add.dataset.n ? `option ${add.dataset.n}, ` : "";
    return send(`Please add ${n}${add.dataset.name}, to my basket.`, "text");
  }
  const confirm = e.target.closest("[data-confirm]");
  if (confirm && !S.busy) {
    confirm.disabled = true;
    renderEntry({ role: "user", text: "Confirm order", channel: "ui" });
    setBusy(true);
    try {
      const r = await post("/api/checkout/confirm", { user_id: S.user, checkout_id: confirm.dataset.confirm });
      S.pendingCheckout = null;
      document.querySelectorAll(".summary .row-actions").forEach((x) => x.remove());
      renderEntry({ role: "assistant", text: r.reply, order: r.order, actions: [{ tool: "confirm_order (app, after you pressed Confirm)", ok: r.ok, summary: r.ok ? `order ${r.order.order_id}` : r.error?.message }] });
      renderBasket(r.basket); refreshPanel();
      if ($("#speak-replies").checked) speak(r.reply);
    } catch (err) { renderEntry({ role: "assistant", text: err.message, error: true }); }
    finally { setBusy(false); }
    return;
  }
  if (e.target.closest("[data-cancel]")) {
    await post("/api/checkout/cancel", { user_id: S.user });
    S.pendingCheckout = null;
    document.querySelectorAll(".summary .row-actions").forEach((x) => x.remove());
    return send("Not yet, I'd like to keep shopping.", "text");
  }
  const ask = e.target.closest("[data-ask]");
  if (ask && !S.busy) return send(ask.dataset.ask, "text");
});

/* ------------------------------------------------------------------ side panel */
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  ["home", "shop", "rewards", "basket"].forEach((t) => { $(`#tab-${t}`).hidden = t !== name; });
  if (name === "shop") loadShop();
  if (name === "rewards") loadRewards();
}
function refreshPanel() {
  loadHome();
  if (!$("#tab-rewards").hidden) loadRewards();
}

function miniProduct(p, extra = "") {
  return `<div class="mini"><div>${esc(p.name)}<div class="meta">${esc((p.category || "").replace(/_/g, " "))}, quality ${p.quality_tier}${p.why?.length ? `, ${esc(p.why[0])}` : ""}</div></div>
    <div class="right">${priceHtml(p)}${extra}</div></div>`;
}

async function loadHome() {
  const [h, m] = await Promise.all([api(`/api/customer/${S.user}/home`), api(`/api/customer/${S.user}/memory`)]);
  const c = h.customer;
  const mem = [];
  if (m.goal) mem.push(`<dt>Goal</dt><dd>${esc(m.goal)}</dd>`);
  if (m.budget) mem.push(`<dt>Budget</dt><dd>${esc(m.budget)}${cur()} ${m.budget_scope === "total" ? "in total" : "per item"}</dd>`);
  if (m.preferences?.length) mem.push(`<dt>Likes</dt><dd>${esc(m.preferences.join(", "))}</dd>`);
  if (m.dislikes?.length) mem.push(`<dt>Avoids</dt><dd>${esc(m.dislikes.join(", "))}</dd>`);
  const rej = Object.keys(m.rejected || {});
  if (rej.length) mem.push(`<dt>Ruled out</dt><dd>${rej.length} product${rej.length > 1 ? "s" : ""}</dd>`);
  $("#tab-home").innerHTML = `
    <div class="block"><h2>About this customer</h2><dl class="kv">
      <dt>Region</dt><dd>${esc(c.region)}</dd><dt>Device</dt><dd>${esc(c.device)}</dd>
      <dt>Household</dt><dd>${esc(c.household_size)} people</dd><dt>Monthly budget</dt><dd>${esc(c.monthly_budget)}${cur()}</dd>
      <dt>Quality</dt><dd>prefers tier ${esc(parseFloat(c.preferred_quality) || "not set")} of 5</dd><dt>Offers</dt><dd>${c.marketing_opt_in ? "opted in" : "not opted in"}</dd>
      ${c.interests.length ? `<dt>Interests</dt><dd>${esc(c.interests.join(", "))}</dd>` : ""}
      ${c.owned.length ? `<dt>Owns</dt><dd>${esc(c.owned.map((o) => o.name).join(", "))}</dd>` : ""}
    </dl></div>
    <div class="block"><h2>What the companion remembers</h2>${mem.length ? `<dl class="kv">${mem.join("")}</dl>` : `<p class="empty">Nothing yet. Goals, budgets and preferences appear here as you talk.</p>`}</div>
    ${h.nudges.length ? `<div><h2>Worth a look</h2>${h.nudges.map((n) => `<p class="nudge">${esc(n.text)}</p>`).join("")}</div>` : ""}
    <div class="block"><h2>Picked for you</h2>${h.for_you.map((p) => miniProduct(p)).join("") || `<p class="empty">No eligible picks right now.</p>`}</div>`;
}

async function loadShop(ev) {
  if (ev) ev.preventDefault();
  const params = new URLSearchParams();
  const q = $("#shop-q").value.trim(), cat = $("#shop-cat").value, max = $("#shop-max").value;
  if (q) params.set("q", q); if (cat) params.set("category", cat); if (max) params.set("max_price", max);
  const r = await api(`/api/customer/${S.user}/shop?${params}`);
  if (!S.categoriesLoaded) {
    const opts = Object.entries(r.categories).map(([d, cats]) => `<optgroup label="${esc(d)}">${Object.keys(cats).map((c) => `<option value="${esc(c)}">${esc(c.replace(/_/g, " "))}</option>`).join("")}</optgroup>`).join("");
    $("#shop-cat").insertAdjacentHTML("beforeend", opts); S.categoriesLoaded = true;
  }
  const hidden = r.hidden_as_ineligible ? `<p class="empty">${Object.values(r.hidden_as_ineligible).reduce((a, b) => a + b, 0)} more match but aren't available for this customer.</p>` : "";
  $("#shop-results").innerHTML = `<div class="block">${r.results.map((p) => miniProduct(p, `<br><button class="btn small" data-ask="Tell me about ${esc(p.name)} and whether it suits me.">Ask</button>`)).join("") || `<p class="empty">No products match. Try a wider price or another category.</p>`}</div>${hidden}`;
}
$("#shop-form").addEventListener("submit", loadShop);

async function loadRewards() {
  const r = await api(`/api/customer/${S.user}/rewards`);
  const act = r.rewards_activity || {};
  $("#tab-rewards").innerHTML = `
    <div class="block"><h2>${esc((r.membership_tier || "").replace(/^./, (x) => x.toUpperCase()))} member</h2>
      <p>${esc(r.tenure_months)} months with us. ${act.events ? `${act.events} visits to Rewards, ${act.redemptions} redemptions.` : "No Rewards activity yet."}</p></div>
    <div class="block"><h2>Offers you can use</h2>${r.offers.map((p) => miniProduct(p)).join("") || `<p class="empty">No discounted items available for this customer right now.</p>`}</div>
    <div class="block"><h2>Orders in this demo</h2>${r.orders.length ? r.orders.slice().reverse().map((o) => `<div class="mini"><div>${esc(o.items.join(", "))}<div class="meta">${esc(o.order_id)}</div></div><div class="right">${esc(o.total)}${cur()}</div></div>`).join("") : `<p class="empty">No orders yet.</p>`}</div>`;
}

function renderBasket(b) {
  $("#basket-count").textContent = b.items.reduce((a, i) => a + i.quantity, 0);
  $("#tab-basket").innerHTML = b.items.length ? `<div class="block">${b.items.map((i) => `
      <div class="mini"><div>${esc(i.product_name)}${i.is_subscription ? ` <span class="badge sub">subscription</span>` : ""}
        <div class="meta">${esc(i.unit_price)}${cur()} each</div>
        <div class="qty"><button data-qty="${esc(i.product_id)}" data-q="${i.quantity - 1}" aria-label="Decrease">−</button>${i.quantity}<button data-qty="${esc(i.product_id)}" data-q="${i.quantity + 1}" aria-label="Increase">+</button>
        <button class="btn small" data-remove="${esc(i.product_id)}">Remove</button></div></div>
        <div class="right"><strong>${esc(i.line_total)}${cur()}</strong></div></div>`).join("")}
      <div class="mini"><strong>Total</strong><strong>${esc(b.total)}${cur()}</strong></div></div>
      <button class="btn primary" data-ask="I'd like to check out.">Check out with the companion</button>`
    : `<p class="empty">The basket is empty. Ask the companion for ideas or browse Shop.</p>`;
}
$("#tab-basket").addEventListener("click", async (e) => {
  const q = e.target.closest("[data-qty]"), r = e.target.closest("[data-remove]");
  if (!q && !r) return;
  const res = q ? await post("/api/basket/update", { user_id: S.user, product_id: q.dataset.qty, quantity: Number(q.dataset.q) })
                : await post("/api/basket/remove", { user_id: S.user, product_id: r.dataset.remove, quantity: 0 });
  if (res.ok) { renderBasket(res.data); S.pendingCheckout = null; document.querySelectorAll(".summary .row-actions").forEach((x) => x.remove()); }
  else alert(res.error.message);
});

/* ------------------------------------------------------------------ voice: capture 16 kHz PCM -> Amazon Transcribe */
const V = { rec: false, stream: null, ac: null, proc: null, src: null, chunks: [], timer: null, audio: null, recog: null };
const status = (t) => { $("#voice-status").textContent = t || ""; };

$("#voice-language").onchange = (e) => {
  S.voiceLanguage = e.target.value;
  if (S.voiceLanguage !== "auto") S.lastDetectedLanguage = S.voiceLanguage;
  status("");
};

$("#btn-mic").onclick = () => {
  if (S.busy && !V.rec) return;
  if (S.sttMode === "browser") return browserRecognition();
  V.rec ? stopRecording() : startRecording();
};

async function startRecording() {
  stopSpeaking();
  try {
    V.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch { status("Microphone blocked. Allow mic access in the browser to talk."); return; }
  V.ac = new (window.AudioContext || window.webkitAudioContext)();
  V.src = V.ac.createMediaStreamSource(V.stream);
  V.proc = V.ac.createScriptProcessor(4096, 1, 1);
  V.chunks = [];
  V.proc.onaudioprocess = (e) => {
    const d = e.inputBuffer.getChannelData(0);
    V.chunks.push(new Float32Array(d));
    let sum = 0; for (let i = 0; i < d.length; i += 16) sum += d[i] * d[i];
    const lvl = Math.min(1, Math.sqrt(sum / (d.length / 16)) * 6);
    $("#mic-ring").style.transform = `scale(${1 + lvl * 0.9})`;
  };
  V.src.connect(V.proc); V.proc.connect(V.ac.destination);
  V.rec = true; $("#btn-mic").classList.add("recording");
  status("Listening. Tap the mic again when you're done.");
  V.timer = setTimeout(stopRecording, 30000);
}

async function stopRecording() {
  clearTimeout(V.timer);
  V.rec = false; $("#btn-mic").classList.remove("recording"); $("#btn-mic").classList.add("busy");
  $("#mic-ring").style.transform = "scale(1)";
  V.src?.disconnect(); V.proc?.disconnect(); V.stream?.getTracks().forEach((t) => t.stop());
  const rate = V.ac.sampleRate; await V.ac.close();
  const pcm = toPcm16(V.chunks, rate, 16000);
  status("Transcribing…");
  try {
    const selected = S.voiceLanguage;
    const res = await fetch(`/api/voice/transcribe?user_id=${encodeURIComponent(S.user)}&rate=16000&language=${encodeURIComponent(selected)}`, { method: "POST", body: pcm, headers: { "Content-Type": "application/octet-stream" } });
    const j = await res.json();
    if (!j.ok) throw new Error(j.error);
    status("");
    if (j.language) S.lastDetectedLanguage = j.language;
    if (j.text) sendVoice(j.text, j.language, j.languages); else status("I didn't catch that. Tap the mic and try again.");
  } catch (e) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) { S.sttMode = "browser"; status("Server speech recognition is unavailable; switched to the browser's. Tap the mic again."); }
    else status(`Voice isn't available right now (${e.message}). Type your message instead.`);
  } finally { $("#btn-mic").classList.remove("busy"); }
}

function toPcm16(chunks, inRate, outRate) {
  const len = chunks.reduce((a, c) => a + c.length, 0);
  const data = new Float32Array(len); let o = 0; for (const c of chunks) { data.set(c, o); o += c.length; }
  const ratio = inRate / outRate, outLen = Math.floor(len / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio), end = Math.min(len, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = start; j < end; j++) s += data[j];
    const v = Math.max(-1, Math.min(1, s / Math.max(1, end - start)));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return out.buffer;
}

function browserRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { status("This browser has no speech recognition. Use Chrome or Edge, or type instead."); return; }
  if (V.recog) { V.recog.stop(); return; }
  const chosen = S.voiceLanguage === "auto" ? S.lastDetectedLanguage : S.voiceLanguage;
  if (!chosen) {
    status("Choose English or العربية once before using browser speech recognition.");
    return;
  }
  stopSpeaking();
  const r = new SR(); V.recog = r;
  r.lang = chosen; r.interimResults = false; r.maxAlternatives = 1;
  $("#btn-mic").classList.add("recording"); status("Listening…");
  r.onresult = (e) => { const t = e.results[0][0].transcript; if (t) sendVoice(t, chosen, [chosen]); };
  r.onerror = (e) => status(`Voice error: ${e.error}`);
  r.onend = () => { V.recog = null; $("#btn-mic").classList.remove("recording"); if ($("#voice-status").textContent === "Listening…") status(""); };
  r.start();
}

/* ------------------------------------------------------------------ voice: Amazon Polly playback */
async function sendVoice(text, language, languages = []) {
  S.lastDetectedLanguage = language || S.lastDetectedLanguage;
  stopSpeaking();
  renderEntry({ role: "user", text, channel: "voice", language, languages });
  setBusy(true);
  try {
    const r = await post("/api/chat", { user_id: S.user, message: text, channel: "voice",
      voice_language: language || null, voice_languages: languages });
    S.pendingCheckout = r.pending_checkout;
    renderEntry(r); renderBasket(r.basket); refreshPanel();
    speak(r.text, r.language);
  } catch (e) {
    renderEntry({ role: "assistant", text: `Something went wrong: ${e.message}. Please try again.`, error: true });
  } finally { setBusy(false); }
}

async function speak(text, language) {
  if (!text) return;
  stopSpeaking();
  if (S.cfg.tts !== "browser") {
    try {
      const blob = await api("/api/voice/speak", { method: "POST", body: JSON.stringify({ text, language }) });
      V.audio = new Audio(URL.createObjectURL(blob));
      await V.audio.play();
      return;
    } catch { /* fall back to the browser voice */ }
  }
  if ("speechSynthesis" in window) {
    const u = new SpeechSynthesisUtterance(text.replace(/[*#_`]/g, "").slice(0, 900));
    u.lang = language || (/[؀-ۿ]/.test(text) ? "ar-SA" : "en-US");
    const voices = speechSynthesis.getVoices();
    const prefix = u.lang.slice(0, 2).toLowerCase();
    u.voice = voices.find((v) => v.lang.toLowerCase() === u.lang.toLowerCase()) ||
      voices.find((v) => v.lang.toLowerCase().startsWith(prefix)) || null;
    speechSynthesis.speak(u);
  }
}
function stopSpeaking() {
  if (V.audio) { V.audio.pause(); V.audio = null; }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
}

boot();
