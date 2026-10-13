// Click the toolbar icon on a Reddit post -> scrape -> download CSV.
const MAX_COMMENTS = 1000;   // change if you want more/less
const MAX_BRANCHES = 15;     // "continue this thread" links to follow

const COLUMNS = ["comment_id","parent_id","depth","author","score",
                 "permalink","text","post_title","post_url","scraped_at"];
const SKIP = new Set(["[deleted]", "[removed]"]);

function csvCell(v) {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function buildCsv(data) {
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);
  const base = { post_title: data.post.title, post_url: data.post.url, scraped_at: now };
  const rows = [];
  if (data.post.text) {
    rows.push({ ...base, comment_id: "post", parent_id: "", depth: -1, author: data.post.author,
                score: data.post.score, permalink: data.post.url, text: data.post.text });
  }
  for (const c of data.comments) {
    const t = (c.text || "").trim();
    if (!t || SKIP.has(t) || c.author === "AutoModerator") continue;
    rows.push({ ...base, comment_id: c.id, parent_id: c.parent_id, depth: c.depth,
                author: c.author, score: c.score, permalink: c.permalink, text: t });
  }
  const lines = [COLUMNS.join(",")];
  for (const r of rows) lines.push(COLUMNS.map((k) => csvCell(r[k])).join(","));
  return { csv: lines.join("\r\n"), n: rows.length };
}

async function badge(tabId, text, color) {
  await chrome.action.setBadgeBackgroundColor({ color, tabId });
  await chrome.action.setBadgeText({ text, tabId });
}

chrome.action.onClicked.addListener(async (tab) => {
  const id = tab.id;
  try {
    if (!/^https:\/\/(www\.)?reddit\.com\/r\/[^/]+\/comments\//.test(tab.url || "")) {
      throw new Error("Open a Reddit post first");
    }
    await badge(id, "...", "#f59e0b");
    await chrome.scripting.executeScript({ target: { tabId: id }, files: ["content.js"] });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: id },
      func: (m, b) => window.__owlScrape(m, b),
      args: [MAX_COMMENTS, MAX_BRANCHES],
    });
    const { csv, n } = buildCsv(result);
    if (n === 0) throw new Error("No comments found");
    const slug = (result.post.id || "post").replace(/[^a-z0-9_-]/gi, "");
    const url = "data:text/csv;charset=utf-8," + encodeURIComponent("\uFEFF" + csv);
    await chrome.downloads.download({ url, filename: `owl_${slug}_${Date.now()}.csv`, saveAs: false });
    await badge(id, String(n), "#16a34a");
  } catch (e) {
    console.error(e);
    await badge(id, "ERR", "#dc2626");
  }
});
