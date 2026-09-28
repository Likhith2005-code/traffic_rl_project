const $ = s => document.querySelector(s);
const AG = ["agent1", "agent2", "agent3"];
const COL = { agent1: "#ff5468", agent2: "#4aa3ff", agent3: "#34d399" };
const ACT = ["Accelerate", "Brake", "Maintain", "Slow", "Fast", "Lane left", "Lane right", "Overtake"];
const TLS = ["Red", "Yellow", "Green"];

let net = null, road = null, S = null, sel = "agent1", zoom = 2, drag = null;
let lastStep = -1, lastEp = -1, lastEv = 0;
const cam = { x: 0, y: 0 }, pos = {}, hist = {}, trail = {};
AG.forEach(a => { hist[a] = []; trail[a] = []; });

/* ---------- static UI ---------- */
const TILES = [
  ["speed", "Speed", "m/s", a => a.speed.toFixed(1)],
  ["gap", "Gap ahead", "m", a => (a.gap >= 500 ? "—" : a.gap.toFixed(0))],
  ["wait", "Waiting", "s", a => a.wait.toFixed(0)],
  ["accel", "Accel", "m/s²", a => a.accel.toFixed(1)],
  ["lane", "Lane", "", a => `${a.lane + 1}/${a.lanes}`],
  ["tls", "Signal", "", a => (a.tls_dist >= 1000 ? "none" : `${TLS[a.tls]} ${a.tls_dist.toFixed(0)}m`)],
  ["act", "Action", "", a => ACT[a.action]],
  ["rew", "Step reward", "", a => a.reward.toFixed(2)],
  ["tot", "Total reward", "", a => a.total.toFixed(0)],
  ["avg", "Avg speed", "m/s", a => a.avg_speed.toFixed(1)],
  ["arr", "Arrivals", "", a => a.arrivals],
  ["col", "Collisions", "", a => a.collisions],
];
$("#tiles").innerHTML = TILES.map(t =>
  `<div class="tile"><small>${t[1]}</small><b id="t_${t[0]}">–</b><i>${t[2]}</i></div>`).join("");
$("#tabs").innerHTML = AG.map((a, i) =>
  `<button class="tab" data-a="${a}" style="color:${COL[a]}"><i></i><b style="color:#e8ecff">${a}</b><span id="tab_${a}">${i + 1}</span></button>`).join("");

function select(a) {
  sel = a;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.a === a));
  $("#hudName").textContent = a; $("#hudName").style.color = COL[a];
  if (S) render();
}
document.querySelectorAll(".tab").forEach(t => t.onclick = () => select(t.dataset.a));
addEventListener("keydown", e => { if (e.key >= "1" && e.key <= "3") select(AG[e.key - 1]); });

/* ---------- controls ---------- */
const post = body => fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body) });
$("#btnPlay").onclick = () => post({ cmd: S && S.running ? "pause" : "start" });
$("#btnReset").onclick = () => post({ cmd: "reset" });
$("#speed").onchange = e => post({ speed: +e.target.value });
$("#det").onchange = e => post({ deterministic: e.target.checked });
$("#zoom").oninput = e => { zoom = +e.target.value; };

/* ---------- data ---------- */
async function loadNet() {
  try {
    const r = await fetch("/api/network");
    if (!r.ok) throw 0;
    net = await r.json();
    road = new Path2D();
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const l of net.lanes) {
      l.p.forEach(([x, y], i) => {
        i ? road.lineTo(x, y) : road.moveTo(x, y);
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      });
    }
    cam.x = (x0 + x1) / 2; cam.y = (y0 + y1) / 2;
  } catch { setTimeout(loadNet, 1000); }
}

async function poll() {
  try {
    const s = await (await fetch("/api/state")).json();
    if (s.ready) onState(s);
    const b = $("#banner");
    b.hidden = !s.error;
    if (s.error) b.textContent = "Simulation error: " + s.error;
  } catch { $("#mode").textContent = "offline – is app.py running?"; }
  setTimeout(poll, 120);
}

function onState(s) {
  S = s;
  if (s.episode !== lastEp) {
    lastEp = s.episode; lastStep = -1;
    AG.forEach(a => { hist[a] = []; trail[a] = []; });
  }
  if (s.step !== lastStep) {
    lastStep = s.step;
    AG.forEach(a => {
      const d = s.agents[a];
      if (d.present) {
        hist[a].push({ v: d.speed, r: d.reward });
        trail[a].push([d.x, d.y]);
        if (hist[a].length > 160) hist[a].shift();
        if (trail[a].length > 70) trail[a].shift();
      }
    });
  }
  const log = $("#log");
  s.events.filter(e => e.id > lastEv).reverse().forEach(e => {
    const li = document.createElement("li");
    li.className = e.kind;
    li.innerHTML = `<span>#${e.step}</span><b>${e.text}</b>`;
    log.prepend(li);
    lastEv = e.id;
  });
  while (log.children.length > 40) log.lastChild.remove();
  render();
}

/* ---------- panels ---------- */
function render() {
  const s = S, a = s.agents[sel];
  $("#mode").textContent = `${s.mode} · episode ${s.episode}`;
  $("#btnPlay").textContent = s.running ? "⏸ Pause" : "▶ Start";
  $("#stepInfo").textContent = `step ${s.step} · ${s.vehicles.length} background vehicles`;
  $("#edge").textContent = a.edge || "";
  TILES.forEach(t => { $("#t_" + t[0]).textContent = a.speed === undefined ? "–" : t[3](a); });
  const p = a.progress || 0;
  $("#prog").style.width = p + "%"; $("#progTxt").textContent = p.toFixed(0) + "%";
  $("#hudInfo").textContent = a.present ? `${a.speed.toFixed(1)} m/s · ${ACT[a.action]}` : "respawning…";
  AG.forEach(k => { const d = s.agents[k]; $("#tab_" + k).textContent = d.present ? d.speed.toFixed(1) + " m/s" : "—"; });
  $("#board").innerHTML = "<tr><th>Agent</th><th>Speed</th><th>Reward</th><th>Arr</th><th>Col</th></tr>" +
    AG.map(k => { const d = s.agents[k];
      return `<tr class="row ${k === sel ? "sel" : ""}" onclick="select('${k}')"><td><span class="dot" style="background:${COL[k]}"></span>${k}</td>` +
        `<td>${d.speed === undefined ? "–" : d.speed.toFixed(1)}</td><td>${d.total.toFixed(0)}</td><td>${d.arrivals}</td><td>${d.collisions}</td></tr>`;
    }).join("");
  drawChart();
}

function drawChart() {
  const c = $("#chart"), d = devicePixelRatio || 1;
  c.width = c.clientWidth * d; c.height = c.clientHeight * d;
  const x = c.getContext("2d"), W = c.width, H = c.height, h = hist[sel];
  x.clearRect(0, 0, W, H);
  x.strokeStyle = "#232d52"; x.lineWidth = d;
  for (let i = 1; i < 4; i++) { x.beginPath(); x.moveTo(0, H * i / 4); x.lineTo(W, H * i / 4); x.stroke(); }
  if (h.length < 2) return;
  const line = (key, color, lo, hi) => {
    x.beginPath(); x.strokeStyle = color; x.lineWidth = 2 * d;
    h.forEach((p, i) => {
      const px = i / (160 - 1) * W, py = H - 6 * d - (p[key] - lo) / ((hi - lo) || 1) * (H - 12 * d);
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    });
    x.stroke();
  };
  const rs = h.map(p => p.r);
  line("v", "#6c8cff", 0, Math.max(15, ...h.map(p => p.v)));
  line("r", "#fbbf24", Math.min(...rs, 0), Math.max(...rs, 1));
}

/* ---------- map renderer ---------- */
const map = $("#map"), ctx = map.getContext("2d");

function car(sx, sy, ang, L, Wd, fill) {
  ctx.save(); ctx.translate(sx, sy); ctx.rotate(ang * Math.PI / 180);
  ctx.fillStyle = fill; ctx.beginPath();
  (ctx.roundRect || ctx.rect).call(ctx, -Wd / 2, 0, Wd, L, Wd * 0.3);
  ctx.fill(); ctx.restore();
}

function frame(t) {
  const d = devicePixelRatio || 1;
  if (map.width !== map.clientWidth * d || map.height !== map.clientHeight * d) {
    map.width = map.clientWidth * d; map.height = map.clientHeight * d;
  }
  const W = map.width, H = map.height, sc = zoom * d;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, W, H);

  if (!net || !S) {
    ctx.fillStyle = "#8b95bd"; ctx.font = `${14 * d}px system-ui`; ctx.textAlign = "center";
    ctx.fillText("Waiting for SUMO simulation…", W / 2, H / 2);
    requestAnimationFrame(frame); return;
  }

  // smooth agent positions + follow camera
  AG.forEach(a => {
    const p = S.agents[a];
    if (!p.present) { delete pos[a]; return; }
    const q = pos[a] || (pos[a] = { x: p.x, y: p.y });
    q.x += (p.x - q.x) * 0.25; q.y += (p.y - q.y) * 0.25; q.a = p.a;
  });
  if ($("#follow").checked && pos[sel]) {
    cam.x += (pos[sel].x - cam.x) * 0.15; cam.y += (pos[sel].y - cam.y) * 0.15;
  }
  const tx = X => (X - cam.x) * sc + W / 2, ty = Y => H / 2 - (Y - cam.y) * sc;

  // roads (world-space Path2D: outline pass then fill pass)
  ctx.setTransform(sc, 0, 0, -sc, W / 2 - cam.x * sc, H / 2 + cam.y * sc);
  ctx.lineJoin = "round"; ctx.lineCap = "butt";
  ctx.strokeStyle = "#2c376a"; ctx.lineWidth = 3.2 + 2 * d / sc; ctx.stroke(road);
  ctx.strokeStyle = "#141b36"; ctx.lineWidth = 3.2; ctx.stroke(road);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // trail of tracked agent
  const tr = trail[sel];
  if ($("#trail").checked && tr.length > 1) {
    ctx.lineWidth = 3 * d; ctx.lineCap = "round";
    for (let i = 1; i < tr.length; i++) {
      ctx.strokeStyle = COL[sel] + Math.round(i / tr.length * 150 + 20).toString(16).padStart(2, "0");
      ctx.beginPath(); ctx.moveTo(tx(tr[i - 1][0]), ty(tr[i - 1][1])); ctx.lineTo(tx(tr[i][0]), ty(tr[i][1])); ctx.stroke();
    }
  }

  // background traffic
  const L = Math.max(4.5 * sc, 6 * d), Wd = Math.max(1.9 * sc, 3 * d);
  for (const [x, y, a] of S.vehicles) car(tx(x), ty(y), a, L, Wd, "#7a86b8");

  // RL agents
  AG.forEach(a => {
    const q = pos[a]; if (!q) return;
    const sx = tx(q.x), sy = ty(q.y);
    ctx.shadowColor = COL[a]; ctx.shadowBlur = a === sel ? 18 * d : 8 * d;
    car(sx, sy, q.a, L * 1.15, Wd * 1.2, COL[a]);
    ctx.shadowBlur = 0;
    if (a === sel) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 * d;
      ctx.beginPath(); ctx.arc(sx, sy, (16 + 3 * Math.sin(t / 250)) * d, 0, 7); ctx.stroke();
    }
    ctx.fillStyle = "#e8ecff"; ctx.font = `${11 * d}px system-ui`; ctx.textAlign = "left";
    ctx.fillText(a, sx + 20 * d, sy - 6 * d);
  });
  requestAnimationFrame(frame);
}

/* manual pan (when Follow is off) + wheel zoom */
map.onmousedown = e => { drag = [e.clientX, e.clientY]; map.style.cursor = "grabbing"; };
addEventListener("mouseup", () => { drag = null; map.style.cursor = "grab"; });
addEventListener("mousemove", e => {
  if (!drag || $("#follow").checked) return;
  cam.x -= (e.clientX - drag[0]) / zoom; cam.y += (e.clientY - drag[1]) / zoom;
  drag = [e.clientX, e.clientY];
});
map.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.min(8, Math.max(0.3, zoom * Math.exp(-e.deltaY * 0.001)));
  $("#zoom").value = zoom;
}, { passive: false });

select("agent1");
loadNet();
poll();
requestAnimationFrame(frame);