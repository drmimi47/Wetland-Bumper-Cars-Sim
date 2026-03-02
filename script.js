'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// script.js  —  top-down floating wetland simulation
// Circular enclosure, buoyant physics, area-uniform spawn distribution.
// Open index.html in a browser; no build step required.
// ─────────────────────────────────────────────────────────────────────────────

const canvas = document.getElementById('sim');
const ctx    = canvas.getContext('2d');

// ── Resize ────────────────────────────────────────────────────────────────────
function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

// ── Circular boundary (re-derived each frame so window resize takes effect) ───
function getBoundary() {
  return {
    cx: canvas.width  / 2,
    cy: canvas.height / 2,
    r:  Math.min(canvas.width, canvas.height) * 0.44,
  };
}

// ── Physics constants ─────────────────────────────────────────────────────────
const DRAG_SMALL  = 0.995;   // drag factor for minimum-radius wetland (least resistance)
const DRAG_LARGE  = 0.985;   // drag factor for maximum-radius wetland (most resistance)
const MAX_SPEED   = 0.8;     // px / frame cap
const RESTITUTION    = 0.22;  // speed fraction kept on object–object collision
const WALL_REST      = 0.18;  // softer restitution at the enclosure wall
const SOLVER_ITERS   = 2;     // collision solver passes — resolves multi-body contact chains
const POS_CORRECTION = 0.8;   // fraction of overlap corrected per pass (< 1 prevents overshoot)
const SLOP           = 0.5;   // px of penetration ignored — stops micro-jitter at rest contact
const DENSITY        = 0.0008;// mass = DENSITY × π × r²
const DRIFT_STR      = 0.00012;// Brownian perturbation strength
const WAVE_STR       = 0.008;  // peak target force for wave noise (not mass-scaled)
const WAVE_LERP      = 0.02;   // EMA blend rate — lower = slower, smoother direction changes
const FLOW_STR       = 0.0006; // current flow force scale (not mass-scaled)
// Each row: [kx, ky, ωx, ωy] — spatial wavenumbers and temporal rates for one eddy octave.
// Three octaves with incommensurate rates prevent periodic repetition.
const FLOW_OCTAVES   = [
  [1.8, 1.5, 0.00028, 0.00019],   // large slow eddy
  [2.9, 2.3, 0.00041, 0.00033],   // mid-scale faster eddy
  [1.2, 2.8, 0.00017, 0.00047],   // wide, slowest drift
];
let flowT       = 0;   // frame counter — drives temporal evolution of the flow field
let flowOffsetX = 0;   // advection offset X (normalised units) — drifts flow lookup position
let flowOffsetY = 0;   // advection offset Y (normalised units)
let   flowVizOn      = false; // click inside enclosure to toggle
let   trailOn        = false; // click outside enclosure to toggle motion trails
const trailClones    = [];    // stored frame snapshots [{x,y,radius}, ...][] — grows until cleared
const TRAIL_SAMPLE   = 2;     // capture a snapshot every N render frames
let   trailTick      = 0;     // render-frame counter for sampling
const SPEED_STEPS    = [1, 1.5, 2, 3, 5];   // available speed multipliers
let   speedIdx       = 0;    // index into SPEED_STEPS
let   simAccum       = 0;    // fractional step accumulator for sub-integer speeds
const COUNT          = 10;   // number of wetlands to simulate
const MIN_R          = 25;
const MAX_R          = 65;

// ── Wetland class ─────────────────────────────────────────────────────────────
class Wetland {
  constructor(x, y, radius) {
    this.x      = x;
    this.y      = y;
    this.vx     = (Math.random() - 0.5) * 0.25;   // tiny random start velocity
    this.vy     = (Math.random() - 0.5) * 0.25;
    this.ax     = 0;
    this.ay     = 0;
    this._fx    = 0;   // net force accumulator X
    this._fy    = 0;   // net force accumulator Y
    this.radius = radius;
    this.mass   = DENSITY * Math.PI * radius * radius;
    this.wxf    = 0;   // smoothed wave-noise force X  (EMA state)
    this.wyf    = 0;   // smoothed wave-noise force Y  (EMA state)
  }

  // ── Step 1 ── clear acceleration + force accumulator
  resetAcceleration() {
    this.ax = this.ay = this._fx = this._fy = 0;
  }

  // ── Step 2 ── accumulate forces from any source
  addForce(fx, fy) {
    this._fx += fx;
    this._fy += fy;
  }

  // ── Step 3 ── a = F / m
  computeAcceleration() {
    this.ax = this._fx / this.mass;
    this.ay = this._fy / this.mass;
  }

  // ── Steps 4–7 ── semi-implicit Euler integration
  integrate() {
    this.vx += this.ax;            // 4. v += a
    this.vy += this.ay;

    // 5. water drag — larger wetlands displace more water and slow down faster
    //    t=0 → MIN_R (lightest drag)   t=1 → MAX_R (heaviest drag)
    const t    = (this.radius - MIN_R) / (MAX_R - MIN_R);
    const drag = DRAG_SMALL + t * (DRAG_LARGE - DRAG_SMALL);
    this.vx *= drag;
    this.vy *= drag;

    const speed = Math.hypot(this.vx, this.vy);   // 6. speed cap
    if (speed > MAX_SPEED) {
      const s = MAX_SPEED / speed;
      this.vx *= s;
      this.vy *= s;
    }

    this.x += this.vx;            // 7. position update
    this.y += this.vy;
  }

  // ── Step 8 ── circular boundary collision
  //   Detection : dist(wetland center, enclosure center) + wetland radius > enclosure radius
  //   Normal    : outward unit vector from enclosure center toward wetland
  //   Correction: push wetland back so it just touches the wall
  //   Reflection: remove outward velocity component, add back a damped fraction inward
  //   Jitter fix: skip if velocity is already pointing inward (vDotN ≤ 0)
  resolveBoundary(b) {
    const dx   = this.x - b.cx;
    const dy   = this.y - b.cy;
    const dist = Math.hypot(dx, dy);
    const maxD = b.r - this.radius;   // max allowed distance from enclosure center

    if (dist < 1e-6 || dist <= maxD) return;   // still inside — nothing to do

    // outward unit normal  (enclosure center → wetland center)
    const nx = dx / dist;
    const ny = dy / dist;

    // positional correction — place wetland exactly at the inner wall surface
    this.x = b.cx + nx * maxD;
    this.y = b.cy + ny * maxD;

    // outward velocity component (positive = moving away from center = bad)
    const vDotN = this.vx * nx + this.vy * ny;

    // jitter guard: only reflect if actually moving outward
    if (vDotN <= 0) return;

    // reflect and damp: new outward component = −WALL_REST × old outward component
    this.vx -= (1 + WALL_REST) * vDotN * nx;
    this.vy -= (1 + WALL_REST) * vDotN * ny;
  }

  // ── Step 10 ── render (top-down aerial — concentric wetland zones)
  draw() {
    const { x, y, radius: r } = this;

    // single circle — transparent fill, thin dark stroke + shadow glow
    ctx.shadowColor  = 'rgba(0, 0, 0, 0.52)';
    ctx.shadowBlur   = 11.5;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle   = 'transparent';
    ctx.fill();
    ctx.strokeStyle = 'hsla(0, 0%, 0%, 0.90)';
    ctx.lineWidth   = 1;
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur  = 0;

    // ── velocity arrow ─────────────────────────────────────────────────────
    // Skip only when velocity is numerically zero — avoids division-by-zero.
    // Using 1e-6 (not a larger threshold) prevents blinking caused by forces
    // randomly cancelling and speed briefly crossing a higher cutoff.
    const speed = Math.hypot(this.vx, this.vy);
    if (speed < 1e-6) return;

    const ux = this.vx / speed;   // unit direction
    const uy = this.vy / speed;

    // Length: base (always visible) + speed-scaled portion.
    // Cap at 92% r so the tip sits just inside the outer ring at max speed.
    const len     = Math.min(r * 0.30 + speed * r * 2.2, r * 0.92);
    const headLen = Math.max(7, len * 0.35);   // arrowhead proportional to shaft

    const tx = x + ux * len;   // tip
    const ty = y + uy * len;

    ctx.strokeStyle = 'hsla(0, 0%, 0%, 0.75)';
    ctx.lineWidth   = 2.5;
    ctx.lineCap     = 'round';

    // shaft
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(tx, ty);
    ctx.stroke();

    // arrowhead — two short lines angled ±30° back from the tip
    const a = Math.atan2(uy, ux);
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - headLen * Math.cos(a - Math.PI / 6),
               ty - headLen * Math.sin(a - Math.PI / 6));
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - headLen * Math.cos(a + Math.PI / 6),
               ty - headLen * Math.sin(a + Math.PI / 6));
    ctx.stroke();

    ctx.lineCap = 'butt';   // restore default
  }
}

// ── Step 9 ── circle–circle collision (positional separation + impulse) ───────
//
//   Two-pass Gauss–Seidel solver:
//     Pass 1 — separate positions + apply velocity impulse
//     Pass 2 — correct residual overlap from multi-body chains
//              (impulse skipped: vRel ≥ 0 guard prevents re-application)
//
//   POS_CORRECTION < 1  prevents over-separation overshoot in dense contact.
//   SLOP            > 0  absorbs micro-penetration from floating-point noise,
//                        stopping nearly-stationary objects from jittering.
//
function resolveCircleCollisions(ws) {
  for (let iter = 0; iter < SOLVER_ITERS; iter++) {
    for (let i = 0; i < ws.length; i++) {
      for (let j = i + 1; j < ws.length; j++) {
        const a = ws[i], b = ws[j];
        const dx   = b.x - a.x;
        const dy   = b.y - a.y;
        const dist = Math.hypot(dx, dy);
        const minD = a.radius + b.radius;

        if (dist >= minD || dist < 1e-6) continue;

        // collision normal (a → b)
        const nx = dx / dist;
        const ny = dy / dist;

        // positional separation — weighted by opposing mass
        // only correct the portion beyond SLOP; scale back by POS_CORRECTION
        const penetration = minD - dist;
        const correction  = Math.max(penetration - SLOP, 0) * POS_CORRECTION;
        const totalM      = a.mass + b.mass;
        a.x -= nx * correction * (b.mass / totalM);
        a.y -= ny * correction * (b.mass / totalM);
        b.x += nx * correction * (a.mass / totalM);
        b.y += ny * correction * (a.mass / totalM);

        // velocity impulse along normal
        // vRel ≥ 0 means already separating — also prevents double-impulse
        // across solver iterations (pass 1 reverses approach; pass 2 sees vRel ≥ 0)
        const vRel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (vRel >= 0) continue;

        const j_imp = (-(1 + RESTITUTION) * vRel) / (1 / a.mass + 1 / b.mass);
        a.vx -= (j_imp / a.mass) * nx;
        a.vy -= (j_imp / a.mass) * ny;
        b.vx += (j_imp / b.mass) * nx;
        b.vy += (j_imp / b.mass) * ny;
      }
    }
  }

  // speed clamp — stacked impulses from simultaneous contacts can briefly
  // exceed MAX_SPEED before frame-level damping catches up
  for (const w of ws) {
    const speed = Math.hypot(w.vx, w.vy);
    if (speed > MAX_SPEED) {
      const s = MAX_SPEED / speed;
      w.vx *= s;
      w.vy *= s;
    }
  }
}

// ── Environmental force 1: Brownian drift (micro-current / wind) ──────────────
function applyDrift(w) {
  const s = DRIFT_STR * w.mass;
  w.addForce((Math.random() - 0.5) * s, (Math.random() - 0.5) * s);
}

// ── Environmental force 2: subtle wave noise ───────────────────────────────────
// Each wetland owns a smoothed force vector (wxf, wyf) that drifts toward a new
// random target every frame using an EMA.  Force is mass-scaled so acceleration
// is uniform across sizes — drag (DRAG_SMALL vs DRAG_LARGE) is what makes
// larger circles slower, not a near-zero force.
//
//   new_wxf = wxf + (random_target − wxf) × WAVE_LERP
//
// With WAVE_LERP = 0.02 the force direction fully rotates over ~50 frames (≈ 0.8 s),
// producing continuously varying, non-periodic, organic motion.
function applyWaveNoise(w) {
  w.wxf += ((Math.random() - 0.5) * WAVE_STR - w.wxf) * WAVE_LERP;
  w.wyf += ((Math.random() - 0.5) * WAVE_STR - w.wyf) * WAVE_LERP;
  w.addForce(w.wxf * w.mass, w.wyf * w.mass);
}

// ── Environmental force 3: spatiotemporal current flow field ──────────────────
//
//   Flow = curl of a sinusoidal scalar potential ψ:
//     ψ = sin(kx·nx + ωx·t) · cos(ky·ny + ωy·t)
//     u = ∂ψ/∂y = −ky · sin(kx·nx + ωx·t) · sin(ky·ny + ωy·t)
//     v = −∂ψ/∂x = −kx · cos(kx·nx + ωx·t) · cos(ky·ny + ωy·t)
//
//   Divergence of the curl is identically zero (∂u/∂x + ∂v/∂y = 0), so there is
//   no long-term accumulation of flow toward any edge or the center.
//
//   Position is normalized by enclosure radius before lookup, so the spatial
//   pattern is independent of window size and fills the whole enclosure.
//   Nearby wetlands share similar forces; distant ones experience independent flow.
//
//   Force is mass-scaled → uniform acceleration across sizes; drag difference
//   (DRAG_SMALL vs DRAG_LARGE) is what makes larger circles cruise slower.
//
function applyCurrentFlow(w, bnd) {
  const nx = (w.x - bnd.cx) / bnd.r + flowOffsetX;   // normalised + advection offset
  const ny = (w.y - bnd.cy) / bnd.r + flowOffsetY;

  let fx = 0, fy = 0;

  for (const [kx, ky, ox, oy] of FLOW_OCTAVES) {
    const phiX = kx * nx + ox * flowT;
    const phiY = ky * ny + oy * flowT;
    fx -= ky * Math.sin(phiX) * Math.sin(phiY);   // ∂ψ/∂y
    fy -= kx * Math.cos(phiX) * Math.cos(phiY);   // −∂ψ/∂x
  }

  const n = FLOW_OCTAVES.length;
  w.addForce((fx / n) * FLOW_STR * w.mass, (fy / n) * FLOW_STR * w.mass);
}

// ── Background + enclosure rendering ─────────────────────────────────────────
function drawScene(b) {
  // new background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // dotted boundary infill (white)
  ctx.beginPath();
  ctx.arc(b.cx, b.cy, b.r, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  // dotted boundary ring
  ctx.shadowColor  = 'rgba(0, 0, 0, 0.45)';
  ctx.shadowBlur   = 10;
  ctx.beginPath();
  ctx.arc(b.cx, b.cy, b.r, 0, Math.PI * 2);
  ctx.setLineDash([16, 9]);  // 6px dash, 9px gap
  ctx.strokeStyle = 'hsla(0, 0%, 0%, 0.90)';
  ctx.lineWidth   = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);   // restore solid for everything else
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur  = 0;

  // speed label — bottom center of boundary ring, clickable
  ctx.font         = '18px helvetica, arial, sans-serif';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle    = 'hsla(0, 0%, 0%, 0.90)';
  ctx.fillText(`${SPEED_STEPS[speedIdx]}x`, b.cx, b.cy + b.r + 18);
  ctx.textBaseline = 'alphabetic';   // restore default
}

// ── Flow field visualization ──────────────────────────────────────────────────
// Samples the same curl-noise field used by applyCurrentFlow at a sparse grid
// of points and draws faint centered line segments aligned with local flow.
// Read-only — no physics state is modified.
function drawFlowViz(b) {
  if (!flowVizOn) return;

  const SPACING = 44;   // px 44 between sample points
  const SEG     = 11;   // 14 half-length of each segment (total = 2 × SEG)

  ctx.lineWidth   = 0.9;
  ctx.lineCap     = 'round';
  ctx.strokeStyle = 'rgba(126, 126, 126, 0.5)';

  const n = FLOW_OCTAVES.length;

  for (let gx = b.cx - b.r; gx <= b.cx + b.r; gx += SPACING) {
    for (let gy = b.cy - b.r; gy <= b.cy + b.r; gy += SPACING) {
      if (Math.hypot(gx - b.cx, gy - b.cy) > b.r - 6) continue;

      // same curl formula as applyCurrentFlow — normalised + advection offset
      const nx = (gx - b.cx) / b.r + flowOffsetX;
      const ny = (gy - b.cy) / b.r + flowOffsetY;
      let fx = 0, fy = 0;
      for (const [kx, ky, ox, oy] of FLOW_OCTAVES) {
        const phiX = kx * nx + ox * flowT;
        const phiY = ky * ny + oy * flowT;
        fx -= ky * Math.sin(phiX) * Math.sin(phiY);
        fy -= kx * Math.cos(phiX) * Math.cos(phiY);
      }
      fx /= n;  fy /= n;

      const mag = Math.hypot(fx, fy);
      if (mag < 1e-6) continue;

      const ux = fx / mag;
      const uy = fy / mag;

      ctx.beginPath();
      ctx.moveTo(gx - ux * SEG, gy - uy * SEG);
      ctx.lineTo(gx + ux * SEG, gy + uy * SEG);
      ctx.stroke();
    }
  }

  ctx.lineCap = 'butt';   // restore default
}

// ── Spawn: area-uniform distribution in disk, no initial overlap ──────────────
//   Using r = maxR × sqrt(U) converts uniform U∈[0,1] to a radius whose PDF is
//   proportional to r — giving equal probability density per unit area throughout
//   the disk, including the centre.
function spawnWetlands() {
  const b    = getBoundary();
  const list = [];

  for (let tries = 0; tries < 5000 && list.length < COUNT; tries++) {
    const r      = MIN_R + Math.random() * (MAX_R - MIN_R);
    const maxRad = b.r - r - 4;                          // keep fully inside wall
    const spawnR = maxRad * Math.sqrt(Math.random());    // area-uniform radial distance
    const theta  = Math.random() * Math.PI * 2;
    const x      = b.cx + spawnR * Math.cos(theta);
    const y      = b.cy + spawnR * Math.sin(theta);

    const clear = list.every(w => Math.hypot(w.x - x, w.y - y) > w.radius + r + 12);
    if (clear) list.push(new Wetland(x, y, r));
  }

  return list;
}

// ── Speed label click + hover cursor ─────────────────────────────────────────
function speedLabelHit(ex, ey) {
  const b  = getBoundary();
  const ly = b.cy + b.r + 10;
  return Math.abs(ex - b.cx) < 28 && ey >= ly && ey <= ly + 16;
}
canvas.addEventListener('click', e => {
  if (speedLabelHit(e.clientX, e.clientY)) {
    speedIdx = (speedIdx + 1) % SPEED_STEPS.length;
    simAccum = 0;
    return;
  }
  const b = getBoundary();
  if (Math.hypot(e.clientX - b.cx, e.clientY - b.cy) <= b.r) {
    flowVizOn = !flowVizOn;
  } else {
    trailOn = !trailOn;
    if (!trailOn) trailClones.length = 0;   // clear all ghosts on toggle-off
  }
});
canvas.addEventListener('mousemove', e => {
  canvas.style.cursor = speedLabelHit(e.clientX, e.clientY) ? 'pointer' : 'default';
});

// ── Spacebar: toggle GIF overlay ─────────────────────────────────────────────
const gifOverlay = document.getElementById('gif-overlay');
document.addEventListener('keydown', e => {
  if (e.code === 'Space') {
    e.preventDefault();
    gifOverlay.classList.toggle('active');
  }
});

// ── Motion trail clones ───────────────────────────────────────────────────────
// Renders stored snapshots oldest→newest with opacity and stroke weight
// scaling linearly from near-invisible (oldest) to near-solid (newest).
function drawTrailClones() {
  if (trailClones.length === 0) return;

  ctx.save();
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.13)';
  ctx.lineWidth   = 0.75;
  for (const frame of trailClones) {
    for (const { x, y, radius } of frame) {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.restore();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
const wetlands = spawnWetlands();

function loop() {
  const b = getBoundary();

  drawScene(b);      // background + enclosure visual

  // Run physics N times this frame to achieve the chosen speed multiplier.
  // simAccum carries fractional remainder so 1.5x alternates 1/2 steps cleanly.
  simAccum += SPEED_STEPS[speedIdx];
  const steps = Math.floor(simAccum);
  simAccum -= steps;

  for (let s = 0; s < steps; s++) {
    flowT++;   // advance flow field time — once per physics step

    // ── Flow field advection ───────────────────────────────────────────────
    // Shift the spatial lookup origin by a slow sinusoidal velocity.
    // Two incommensurate frequencies per axis → non-repeating, non-constant.
    // Integral of sin is bounded (-cos), so the offset never accumulates
    // indefinitely — vortices wander but do not drift off to one side forever.
    // Max offset ≈ ±0.37 normalised units; period ≈ 3–5 min at 60 fps.
    flowOffsetX += 0.000095 * Math.sin(0.000523 * flowT)
                 + 0.000065 * Math.sin(0.000349 * flowT);
    flowOffsetY += 0.000095 * Math.sin(0.000419 * flowT + 1.91)
                 + 0.000065 * Math.sin(0.000277 * flowT + 0.85);

    for (const w of wetlands) {
      w.resetAcceleration();   // 1
      applyDrift(w);           // 2a Brownian micro-current
      applyWaveNoise(w);       // 2b wave noise (EMA-smoothed)
      applyCurrentFlow(w, b);  // 2c spatiotemporal current flow (curl noise)
      w.computeAcceleration(); // 3  a = F / m
      w.integrate();           // 4  v += a  |  5  drag  |  6  clamp  |  7  pos
      w.resolveBoundary(b);    // 8  circular wall collision
    }

    resolveCircleCollisions(wetlands);  // 9  object–object collision
  }

  // ── Trail snapshot capture ─────────────────────────────────────────────────
  // Sample every TRAIL_SAMPLE render frames when trail mode is active.
  // Clones are plain data objects — no physics, just frozen position + radius.
  if (trailOn) {
    trailTick++;
    if (trailTick % TRAIL_SAMPLE === 0) {
      trailClones.push(wetlands.map(w => ({ x: w.x, y: w.y, radius: w.radius })));
    }
  }

  drawTrailClones();           // render stored clones
  drawFlowViz(b);              // flow field overlay on top of clones (click inside enclosure)

  for (const w of wetlands) {
    w.draw();                  // 10 render (active circles on top)
  }

  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
