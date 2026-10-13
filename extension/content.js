// Injected into the Reddit tab. Defines window.__owlScrape(maxComments, maxBranches)
(() => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const MORE_RE = /(more repl|more comment|view more comment)/i;
  const count = (root = document) => root.querySelectorAll("shreddit-comment").length;

  function overlay(text) {
    let el = document.getElementById("__owl_status");
    if (!el) {
      el = document.createElement("div");
      el.id = "__owl_status";
      el.style.cssText = "position:fixed;bottom:16px;right:16px;z-index:999999;background:#111;color:#fff;padding:10px 14px;border-radius:8px;font:14px sans-serif;box-shadow:0 2px 8px #0008";
      document.body.appendChild(el);
    }
    el.textContent = "Opinion Owl: " + text;
  }

  function clickMore() {
    let n = 0;
    document.querySelectorAll("button").forEach((b) => {
      if (MORE_RE.test(b.innerText || "") && b.offsetParent !== null) { b.click(); n++; }
    });
    return n;
  }

  async function waitGrow(before, ms) {
    const end = Date.now() + ms;
    while (Date.now() < end) {
      if (count() > before) return true;
      await sleep(100);
    }
    return false;
  }

  function normId(id) {
    if (!id) return id;
    return /^t[13]_/.test(id) ? id : "t1_" + id;
  }

  function extract(root, limit) {
    const out = [];
    for (const c of root.querySelectorAll("shreddit-comment")) {
      const body = c.querySelector('[slot="comment"]');
      const text = body ? (body.innerText || body.textContent || "").trim() : "";
      if (!text) continue;
      const pid = normId(c.getAttribute("parentid"));
      const depth = c.getAttribute("depth");
      const perma = c.getAttribute("permalink");
      out.push({
        id: normId(c.getAttribute("thingid")),
        parent_id: pid && pid.startsWith("t3_") ? "" : (pid || ""),
        depth: depth !== null && /^\d+$/.test(depth) ? Number(depth) : "",
        author: c.getAttribute("author") || "",
        score: c.getAttribute("score") || "",
        permalink: perma ? "https://reddit.com" + perma : "",
        text,
      });
      if (out.length >= limit) break;
    }
    return out;
  }

  function contLinks(root) {
    return Array.from(root.querySelectorAll("a"))
      .filter((a) => /continue this thread/i.test(a.textContent || ""))
      .map((a) => new URL(a.getAttribute("href"), location.origin).href);
  }

  async function fetchBranch(url, limit) {
    try {
      const res = await fetch(url, { credentials: "include" });
      if (!res.ok) return { comments: [], links: [] };
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      return { comments: extract(doc, limit), links: contLinks(doc) };
    } catch (e) {
      return { comments: [], links: [] };
    }
  }

  window.__owlScrape = async (maxComments = 1000, maxBranches = 15) => {
    const postEl = document.querySelector("shreddit-post");
    if (!postEl) throw new Error("Open a Reddit post page (new Reddit, not old.reddit) first.");

    overlay("expanding...");
    clickMore();
    let prev = -1, stable = 0;
    for (let i = 0; i < 60; i++) {
      const before = count();
      window.scrollTo(0, document.body.scrollHeight);
      await waitGrow(before, 1500);
      const clicked = clickMore();
      if (clicked) await waitGrow(count(), 2000);
      const cur = count();
      overlay(cur + " comments loaded");
      if (cur >= maxComments) break;
      stable = cur === prev && !clicked ? stable + 1 : 0;
      if (stable >= 3) break;
      prev = cur;
    }

    let flat = extract(document, maxComments);
    const seen = new Set(contLinks(document));
    const pending = Array.from(seen);
    let done = 0;
    while (pending.length && done < maxBranches && flat.length < maxComments) {
      const batch = pending.splice(0, Math.min(3, maxBranches - done));
      done += batch.length;
      overlay("following " + done + " deep branch(es)...");
      const results = await Promise.all(batch.map((u) => fetchBranch(u, maxComments - flat.length)));
      for (const r of results) {
        flat = flat.concat(r.comments);
        for (const l of r.links) if (!seen.has(l)) { seen.add(l); pending.push(l); }
      }
    }

    const byId = new Map();
    for (const c of flat) byId.set(c.id, c);
    flat = Array.from(byId.values()).slice(0, maxComments);

    const bodyEl = postEl.querySelector("shreddit-post-text-body, [slot='text-body']");
    const post = {
      id: postEl.getAttribute("id") || postEl.getAttribute("post-id") || "",
      title: postEl.getAttribute("post-title") || document.title,
      author: postEl.getAttribute("author") || "",
      score: postEl.getAttribute("score") || "",
      text: bodyEl ? (bodyEl.innerText || "").trim() : "",
      url: location.href,
    };
    overlay("done - " + flat.length + " comments");
    setTimeout(() => document.getElementById("__owl_status")?.remove(), 3000);
    return { post, comments: flat, branches: done };
  };
})();
