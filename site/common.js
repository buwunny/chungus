// Shared by every page: the registry's address, HTTPS, and loading the model index.
"use strict";

(function () {
  const local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  // Pages and the VPS both redirect to HTTPS on their own; this covers anywhere else.
  if (location.protocol === "http:" && !local) {
    location.replace("https://" + location.host + location.pathname + location.search + location.hash);
  }

  const config = window.CHUNGUS || {};
  let registry = (config.registry || "").replace(/\/+$/, "");
  // Only talk to a registry over HTTPS (plain HTTP is allowed for local testing).
  if (registry && !/^https:\/\//.test(registry) && !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(registry)) {
    console.error(`ignoring registry ${registry}: it must be https://`);
    registry = "";
  }

  async function getJSON(url) {
    const resp = await fetch(url, { cache: "no-cache" });
    if (!resp.ok) throw new Error(`${url}: ${resp.status}`);
    return resp.json();
  }

  // The published models grouped by name, newest first, and the log's signed head.
  // Falls back to the snapshot deployed with the site when the registry is unreachable.
  async function loadIndex() {
    let index, head;
    try {
      index = await getJSON(`${registry}/v1/index`);
      head = await getJSON(`${registry}/v1/head`).catch(() => null);
    } catch (e) {
      index = await getJSON("index.json");
      head = await getJSON("head.json").catch(() => null);
    }
    const byName = new Map();
    for (const hit of index) {
      if (!byName.has(hit.name)) byName.set(hit.name, { name: hit.name, revs: [] });
      byName.get(hit.name).revs.push(hit);
    }
    const models = [...byName.values()];
    for (const m of models) m.revs.sort((a, b) => b.time - a.time);
    models.sort((a, b) => b.revs[0].time - a.revs[0].time);
    return { models, head };
  }

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
  }

  function ago(secs) {
    const d = Date.now() / 1000 - secs;
    const units = [[31536000, "year"], [2592000, "month"], [86400, "day"], [3600, "hour"], [60, "minute"]];
    for (const [n, name] of units) {
      const v = Math.floor(d / n);
      if (v >= 1) return `${v} ${name}${v > 1 ? "s" : ""} ago`;
    }
    return "just now";
  }

  function short(key) {
    return key.length > 20 ? key.slice(0, 16) + "…" + key.slice(-4) : key;
  }

  function command(text) {
    const button = el("button", { type: "button", text: "Copy" });
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select it";
      }
      setTimeout(() => (button.textContent = "Copy"), 1500);
    });
    return el("div", { class: "cmd" }, el("pre", {}, el("code", { text: text })), button);
  }

  function logLine(head) {
    if (!head) return "";
    return (
      `The log has ${head.size} entr${head.size === 1 ? "y" : "ies"}, signed by operator ${head.signature.key}. ` +
      `Check it yourself with: chungus audit --operator ${head.signature.key}`
    );
  }

  window.chungus = {
    registryUrl: registry || location.origin,
    loadIndex,
    el,
    ago,
    short,
    command,
    logLine,
  };
})();
