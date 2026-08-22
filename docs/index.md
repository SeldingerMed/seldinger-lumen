---
title: Lumen | Endovascular simulation
---

<style>
  @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap");

  :root {
    --bg: #f1f1ee;
    --ink: #0c0c0c;
    --graphite: #50504d;
    --grey: #e7e6e0;
    --amber: #e45c24;
    --rule: rgba(12, 12, 12, 0.18);
  }

  html {
    background: var(--bg);
  }

  body {
    background: var(--bg);
    color: var(--ink);
  }

  .page-header,
  .site-footer {
    display: none;
  }

  .main-content {
    max-width: none;
    padding: 0;
  }

  .main-content :is(h1, h2, h3, p, dl, pre, figure) {
    margin-top: 0;
  }

  .lumen-page {
    min-height: 100vh;
    background: var(--bg);
    color: var(--ink);
    font-family: "Newsreader", Georgia, serif;
    margin: 0 calc(50% - 50vw);
  }

  .lumen-page *,
  .lumen-page *::before,
  .lumen-page *::after {
    box-sizing: border-box;
  }

  .lumen-shell {
    width: min(1180px, calc(100% - 3rem));
    margin: 0 auto;
  }

  .lumen-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 4.5rem;
    border-bottom: 1px solid var(--rule);
  }

  .lumen-wordmark {
    color: var(--ink);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .lumen-wordmark span {
    color: var(--amber);
  }

  .lumen-nav-links {
    display: flex;
    gap: 1.4rem;
  }

  .lumen-nav a,
  .lumen-page a {
    color: inherit;
    text-decoration: none;
  }

  .lumen-nav-links a {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .lumen-nav-links a:hover,
  .research-link:hover .research-action {
    color: var(--amber);
  }

  .section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--graphite);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .section-label span {
    color: var(--amber);
  }

  .hero {
    display: grid;
    grid-template-columns: minmax(0, 0.82fr) minmax(420px, 1fr);
    gap: clamp(2rem, 6vw, 6rem);
    align-items: center;
    padding: clamp(5rem, 10vw, 9rem) 0;
  }

  .hero h1 {
    max-width: 11ch;
    margin: 1.6rem 0 1.8rem;
    color: var(--ink);
    font-family: "Newsreader", Georgia, serif;
    font-size: clamp(4rem, 8vw, 7.6rem);
    font-weight: 500;
    letter-spacing: -0.055em;
    line-height: 0.86;
  }

  .hero h1 em {
    color: var(--amber);
    font-weight: 400;
  }

  .hero-copy {
    max-width: 35rem;
    color: var(--graphite);
    font-size: clamp(1.2rem, 2vw, 1.55rem);
    line-height: 1.45;
  }

  .hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 2.2rem;
  }

  .button {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    min-width: 10rem;
    border: 1px solid var(--ink);
    padding: 0.9rem 1rem;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    transition: background-color 160ms ease, color 160ms ease;
  }

  .button.primary,
  .button:hover {
    background: var(--ink);
    color: var(--bg);
  }

  .button.primary:hover {
    background: var(--amber);
    border-color: var(--amber);
    color: var(--ink);
  }

  .hero-media {
    position: relative;
    border: 1px solid var(--ink);
    background: var(--ink);
    padding: 0.45rem;
  }

  .hero-media::before {
    content: "LIVE / SIMULATION";
    position: absolute;
    top: -1.9rem;
    right: 0;
    color: var(--graphite);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
  }

  .hero-media::after {
    content: "";
    position: absolute;
    width: 0.55rem;
    height: 0.55rem;
    top: -0.3rem;
    left: -0.3rem;
    background: var(--amber);
  }

  .hero-media video,
  .capture img {
    display: block;
    width: 100%;
    height: auto;
  }

  .benchmark {
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    background: var(--grey);
  }

  .benchmark-inner {
    display: grid;
    grid-template-columns: 0.75fr 1.25fr;
    gap: clamp(2rem, 7vw, 7rem);
    padding: clamp(4rem, 8vw, 7rem) 0;
  }

  .benchmark h2,
  .section-heading {
    margin: 1.4rem 0 0;
    color: var(--ink);
    font-family: "Newsreader", Georgia, serif;
    font-size: clamp(2.8rem, 5.5vw, 5.4rem);
    font-weight: 500;
    letter-spacing: -0.045em;
    line-height: 0.95;
  }

  .benchmark-copy {
    color: var(--graphite);
    font-size: 1.25rem;
    line-height: 1.55;
  }

  .metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    margin-top: 2.5rem;
    border-top: 1px solid var(--rule);
  }

  .metric {
    padding: 1.3rem 1rem 0 0;
    border-right: 1px solid var(--rule);
  }

  .metric:last-child {
    border-right: 0;
    padding-left: 1rem;
  }

  .metric:nth-child(2) {
    padding-left: 1rem;
  }

  .metric strong {
    display: block;
    color: var(--amber);
    font-family: "Newsreader", Georgia, serif;
    font-size: clamp(2rem, 4vw, 3.8rem);
    font-weight: 500;
    line-height: 1;
  }

  .metric span {
    display: block;
    margin-top: 0.55rem;
    color: var(--graphite);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.61rem;
    line-height: 1.45;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .content-section {
    padding: clamp(5rem, 10vw, 9rem) 0;
    border-bottom: 1px solid var(--rule);
  }

  .section-intro {
    display: grid;
    grid-template-columns: 0.72fr 1.28fr;
    gap: clamp(2rem, 7vw, 7rem);
    align-items: start;
    margin-bottom: 3.5rem;
  }

  .section-intro p {
    max-width: 42rem;
    color: var(--graphite);
    font-size: 1.25rem;
    line-height: 1.55;
  }

  .capability-list {
    border-top: 1px solid var(--ink);
  }

  .capability-list > div {
    display: grid;
    grid-template-columns: 2.4rem minmax(12rem, 0.65fr) 1.35fr;
    gap: 1.5rem;
    padding: 1.35rem 0;
    border-bottom: 1px solid var(--rule);
  }

  .capability-list .index,
  .capability-list dt,
  .capture figcaption,
  .research-kind,
  .research-action {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .capability-list .index {
    color: var(--amber);
  }

  .capability-list dt {
    color: var(--ink);
    font-weight: 500;
  }

  .capability-list dd {
    margin: 0;
    color: var(--graphite);
    font-size: 1.05rem;
    line-height: 1.45;
  }

  .capture-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.5rem;
  }

  .capture {
    margin: 0;
    border-top: 1px solid var(--ink);
    padding-top: 0.5rem;
  }

  .capture img {
    aspect-ratio: 16 / 10;
    object-fit: cover;
    background: var(--ink);
  }

  .capture figcaption {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 0.65rem;
    color: var(--graphite);
  }

  .run-grid {
    display: grid;
    grid-template-columns: 0.72fr 1.28fr;
    gap: clamp(2rem, 7vw, 7rem);
    align-items: start;
  }

  .code-block {
    margin: 0;
    border: 1px solid var(--ink);
    background: var(--ink);
    color: var(--bg);
    padding: clamp(1.2rem, 3vw, 2rem);
    overflow-x: auto;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.78rem;
    line-height: 1.8;
  }

  .research-list {
    border-top: 1px solid var(--ink);
  }

  .research-link {
    display: grid;
    grid-template-columns: 8rem 1fr auto;
    gap: 1.5rem;
    align-items: baseline;
    padding: 1.35rem 0;
    border-bottom: 1px solid var(--rule);
  }

  .research-kind,
  .research-action {
    color: var(--graphite);
  }

  .research-title {
    font-size: 1.2rem;
  }

  .lumen-footer {
    display: flex;
    justify-content: space-between;
    gap: 2rem;
    padding: 2rem 0 3rem;
    color: var(--graphite);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .lumen-footer a {
    color: var(--amber);
  }

  .lumen-page a:focus-visible {
    outline: 2px solid var(--amber);
    outline-offset: 4px;
  }

  @media (max-width: 820px) {
    .hero,
    .benchmark-inner,
    .section-intro,
    .run-grid {
      grid-template-columns: 1fr;
    }

    .hero {
      padding-top: 4rem;
    }

    .hero-media {
      margin-top: 1.5rem;
    }

    .capability-list > div {
      grid-template-columns: 2rem 1fr;
    }

    .capability-list dd {
      grid-column: 2;
    }
  }

  @media (max-width: 600px) {
    .lumen-shell {
      width: min(100% - 2rem, 1180px);
    }

    .lumen-nav {
      align-items: flex-start;
      padding: 1.1rem 0;
    }

    .lumen-nav-links {
      display: grid;
      gap: 0.35rem;
      text-align: right;
    }

    .hero h1 {
      font-size: clamp(3.3rem, 19vw, 5.2rem);
    }

    .metric-row,
    .capture-grid {
      grid-template-columns: 1fr;
    }

    .metric {
      border-right: 0;
      border-bottom: 1px solid var(--rule);
      padding: 1rem 0;
    }

    .metric:nth-child(2),
    .metric:last-child {
      padding-left: 0;
    }

    .research-link {
      grid-template-columns: 1fr auto;
      gap: 0.5rem 1rem;
    }

    .research-kind {
      grid-column: 1 / -1;
    }
    .lumen-footer {
      display: grid;
      gap: 0.65rem;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .lumen-page * {
      scroll-behavior: auto !important;
      transition-duration: 0.001ms !important;
    }
  }
</style>

<div class="lumen-page">
  <div class="lumen-shell">
    <nav class="lumen-nav" aria-label="Primary">
      <a class="lumen-wordmark" href="https://seldinger.med">Seldinger <span>/</span> Lumen</a>
      <div class="lumen-nav-links">
        <a href="https://github.com/SeldingerMed/seldinger-lumen">GitHub</a>
        <a href="assets/launch/lumen-preprint.pdf">Preprint</a>
        <a href="#run">Install</a>
      </div>
    </nav>

    <section class="hero">
      <div>
        <div class="section-label"><span>01</span> Endovascular simulation</div>
        <h1>Target reached. <em>Wall respected.</em></h1>
        <p class="hero-copy">Train catheter and guidewire policies in deformable vascular anatomy. Lumen separates target reach from wall-safe reach, renders matched imaging, and records every episode for replay.</p>
        <div class="hero-actions">
          <a class="button primary" href="https://github.com/SeldingerMed/seldinger-lumen">Run Lumen <span>↗</span></a>
          <a class="button" href="assets/launch/lumen-preprint.pdf">Read preprint <span>↓</span></a>
        </div>
      </div>
      <figure class="hero-media">
        <video src="assets/launch/lumen-launch.mp4" poster="assets/launch/physics-layer.png" controls muted playsinline preload="metadata"></video>
      </figure>
    </section>
  </div>

  <section class="benchmark">
    <div class="lumen-shell benchmark-inner">
      <div>
        <div class="section-label"><span>02</span> Matched benchmark</div>
        <h2>Reach is only half the score.</h2>
      </div>
      <div>
        <p class="benchmark-copy">In a matched branch-navigation PPO run, both environments trained for 50,000 steps and were evaluated for 30 deterministic held-out episodes. Lumen reached 100% raw success and 100% safe success on <code>nav_tree_branch</code>. CathSim reached 100% raw success on <code>phantom3_bca</code>, but 6.7% safe success under the comparison force threshold.</p>
        <div class="metric-row" aria-label="Matched benchmark results">
          <div class="metric"><strong>100%</strong><span>Lumen safe success<br>30 eval episodes</span></div>
          <div class="metric"><strong>6.7%</strong><span>CathSim safe success<br>Matched threshold</span></div>
          <div class="metric"><strong>6.6×</strong><span>Evaluation throughput<br>79.7 vs 12.1 steps/s</span></div>
        </div>
      </div>
    </div>
  </section>

  <div class="lumen-shell">
    <section class="content-section" id="systems">
      <div class="section-intro">
        <div>
          <div class="section-label"><span>03</span> System</div>
          <h2 class="section-heading">Shared simulation state.</h2>
        </div>
        <p>The solver, sensors, observations, and episode record share one state. Change the anatomy, device, or sensor without rebuilding the stack.</p>
      </div>
      <dl class="capability-list">
        <div><span class="index">01</span><dt>Mechanics</dt><dd>Fixed-port guidewire and catheter actuation, deformable walls, finite-radius contact, torsion, friction, flow, clot, retrieval, and flow diversion.</dd></div>
        <div><span class="index">02</span><dt>Imaging</dt><dd>Fluoroscopy, luminal RGB, masks, keypoints, detector noise, dose, latency, and dropout generated from the same case.</dd></div>
        <div><span class="index">03</span><dt>Training</dt><dd>Gymnasium environments, vector execution, privileged state, tracked observations, and recurrent raw-image policies.</dd></div>
        <div><span class="index">04</span><dt>Evidence</dt><dd>Validated episodes, deterministic splits, signed benchmark evidence, replay checks, confidence intervals, and failure taxonomy.</dd></div>
      </dl>
    </section>

    <section class="content-section">
      <div class="section-intro">
        <div>
          <div class="section-label"><span>04</span> Captures</div>
          <h2 class="section-heading">Simulator output.</h2>
        </div>
        <p>Every image below comes from the public simulator and ships with the launch package.</p>
      </div>
      <div class="capture-grid">
        <figure class="capture"><img src="assets/launch/sensor-layer.png" alt="Lumen fluoroscopy, masks, and luminal sensor outputs"><figcaption><span>Sensor layer</span><span>Multimodal</span></figcaption></figure>
        <figure class="capture"><img src="assets/launch/physics-layer.png" alt="Lumen vessel, device, clot, and flow state"><figcaption><span>Physics layer</span><span>State</span></figcaption></figure>
        <figure class="capture"><img src="assets/launch/nav-frame.png" alt="Lumen navigation rollout in branching anatomy"><figcaption><span>Navigation</span><span>Policy rollout</span></figcaption></figure>
        <figure class="capture"><img src="assets/launch/benchmark-outro.png" alt="Lumen matched benchmark summary"><figcaption><span>Evaluation</span><span>Matched run</span></figcaption></figure>
      </div>
    </section>

    <section class="content-section" id="run">
      <div class="run-grid">
        <div>
          <div class="section-label"><span>05</span> Run</div>
          <h2 class="section-heading">Install and run.</h2>
        </div>
        <pre class="code-block"><code>git clone https://github.com/SeldingerMed/seldinger-lumen
cd seldinger-lumen
pip install -e ".[dev]"

lumen doctor
lumen play stenotic --out lumen-run
lumen benchmark lumen-bench
lumen capture lumen-episodes</code></pre>
      </div>
    </section>

    <section class="content-section">
      <div class="section-intro">
        <div>
          <div class="section-label"><span>06</span> Research package</div>
          <h2 class="section-heading">Files and results.</h2>
        </div>
        <p>Read the methods, inspect the benchmark tables, and reproduce the run from the public code.</p>
      </div>
      <div class="research-list">
        <a class="research-link" href="assets/launch/lumen-preprint.pdf"><span class="research-kind">Paper</span><span class="research-title">Lumen launch preprint</span><span class="research-action">PDF ↓</span></a>
        <a class="research-link" href="assets/launch/lumen-preprint-latex.zip"><span class="research-kind">Source</span><span class="research-title">Preprint LaTeX package</span><span class="research-action">ZIP ↓</span></a>
        <a class="research-link" href="assets/launch/benchmark/ppo-short-50k-lumen-cathsim-summary.csv"><span class="research-kind">Benchmark</span><span class="research-title">Matched PPO summary</span><span class="research-action">CSV ↓</span></a>
        <a class="research-link" href="assets/launch/benchmark/pilot-summary-lumen-cathsim-steve.csv"><span class="research-kind">Pilot</span><span class="research-title">Lumen, CathSim, and stEVE summary</span><span class="research-action">CSV ↓</span></a>
        <a class="research-link" href="https://github.com/SeldingerMed/seldinger-lumen"><span class="research-kind">Code</span><span class="research-title">Public repository</span><span class="research-action">GitHub ↗</span></a>
      </div>
    </section>

    <footer class="lumen-footer">
      <span>Apache-2.0 · Seldinger open research</span>
      <a href="https://seldinger.med">Seldinger.med ↗</a>
    </footer>
  </div>
</div>
