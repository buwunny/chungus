// Where the site reads models from. Leave `registry` empty when the site is served by
// the registry's own web server (deploy/): it then uses /v1 on the same origin. On
// GitHub Pages the deploy workflow sets it to the public registry's URL. If the registry
// can't be reached, the site falls back to index.json, a snapshot taken at deploy time.
window.CHUNGUS = { registry: "" };
