---
title: Lumen
---

<style>
  .page-header {
    display: none;
  }
  .main-content {
    max-width: 1120px;
    padding-top: 2rem;
  }
  .lumen-page {
    --ink: #edf4f8;
    --muted: #aebbc7;
    --line: #344154;
    --panel: #111720;
    --bg: #090b10;
    --cyan: #68d7e1;
    --green: #70d38d;
    color: var(--ink);
    background:
      linear-gradient(rgba(104, 215, 225, 0.055) 1px, transparent 1px),
      linear-gradient(90deg, rgba(104, 215, 225, 0.045) 1px, transparent 1px),
      var(--bg);
    background-size: 64px 64px;
    margin: -2rem calc(50% - 50vw) 0;
    padding: 3.2rem max(1.4rem, calc(50vw - 560px)) 4rem;
  }
  .lumen-kicker {
    color: var(--cyan);
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: .8rem;
  }
  .lumen-hero {
    display: grid;
    grid-template-columns: minmax(0, .72fr) minmax(360px, 1fr);
    gap: 2rem;
    align-items: center;
  }
  .lumen-hero h1 {
    color: var(--ink);
    font-size: clamp(2.8rem, 7vw, 5.8rem);
    line-height: .95;
    letter-spacing: 0;
    margin: 0 0 1.1rem;
  }
  .lumen-lede {
    color: #d6dee8;
    font-size: 1.16rem;
    line-height: 1.55;
    margin: 0;
  }
  .lumen-actions {
    display: flex;
    flex-wrap: wrap;
    gap: .75rem;
    margin: 1.35rem 0 0;
  }
  .lumen-button {
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--ink);
    display: inline-block;
    font-weight: 800;
    padding: .76rem .95rem;
  }
  .lumen-button.primary {
    background: var(--cyan);
    border-color: var(--cyan);
    color: #071015;
  }
  .lumen-media {
    border: 1px solid var(--line);
    background: #05080c;
  }
  .lumen-media img,
  .lumen-media video {
    display: block;
    width: 100%;
    height: auto;
  }
  .section {
    border-top: 1px solid var(--line);
    margin-top: 2.6rem;
    padding-top: 2rem;
  }
  .section h2 {
    color: var(--ink);
    font-size: clamp(1.8rem, 4vw, 3rem);
    line-height: 1;
    letter-spacing: 0;
    margin: 0 0 1rem;
  }
  .section p,
  .section li {
    color: #d6dee8;
    font-size: 1.02rem;
    line-height: 1.55;
  }
  .grid-2,
  .grid-3 {
    display: grid;
    gap: 1rem;
  }
  .grid-2 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .grid-3 {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .metric {
    border: 1px solid var(--line);
    background: rgba(17, 23, 32, .82);
    padding: 1rem;
  }
  .metric strong {
    color: var(--green);
    display: block;
    font-size: 1.05rem;
    margin-bottom: .3rem;
  }
  .code-block {
    background: #05080c;
    border: 1px solid var(--line);
    color: #d6dee8;
    overflow-x: auto;
    padding: 1rem;
  }
  @media (max-width: 820px) {
    .lumen-hero,
    .grid-2,
    .grid-3 {
      grid-template-columns: 1fr;
    }
  }
</style>

<div class="lumen-page">
  <section class="lumen-hero">
    <div>
      <div class="lumen-kicker">Open endovascular AI environment</div>
      <h1>Wall-safe vascular navigation, in the open.</h1>
      <p class="lumen-lede">
        Lumen is an Apache-2.0 simulator for training and evaluating endovascular AI agents across deformable vascular anatomy, tube-intrinsic contact, synthetic fluoroscopy, luminal RGB, CV labels, and safety-scored Gymnasium benchmarks.
      </p>
      <div class="lumen-actions">
        <a class="lumen-button primary" href="https://github.com/SeldingerMed/seldinger-lumen">GitHub</a>
        <a class="lumen-button" href="assets/launch/lumen-preprint.pdf">Preprint PDF</a>
        <a class="lumen-button" href="assets/launch/lumen-launch.mp4">Launch video</a>
      </div>
    </div>
    <div class="lumen-media">
      <img src="assets/launch/hero-frame.png" alt="Lumen wall-safe endovascular navigation rollout">
    </div>
  </section>

  <section class="section">
    <div class="lumen-media">
      <video controls playsinline poster="assets/launch/social-card.png">
        <source src="assets/launch/lumen-launch.mp4" type="video/mp4">
      </video>
    </div>
  </section>

  <section class="section">
    <h2>What Lumen Solves</h2>
    <div class="grid-3">
      <div class="metric"><strong>Wall safety is scored</strong> Target reach is separated from safe target reach, so unsafe wall interaction does not look like a clean success.</div>
      <div class="metric"><strong>The lumen is state</strong> Contact, route progress, wall penetration, torsion, and friction hooks are emitted from the same simulation stack.</div>
      <div class="metric"><strong>Images are first-class</strong> Fluoroscopy, masks, keypoints, labels, detector noise, and luminal RGB are generated from one scene.</div>
      <div class="metric"><strong>Advanced use cases ship</strong> Flow diversion, aneurysm inflow traces, clot fields, retrieval, and fragmentation are exposed as simulator state.</div>
      <div class="metric"><strong>Benchmarks are reproducible</strong> Cases, captures, episode sidecars, indexes, and splits are designed for reruns and comparison.</div>
      <div class="metric"><strong>The stack is public</strong> The repository, launch video, screenshots, and preprint are available from this page.</div>
    </div>
  </section>

  <section class="section">
    <h2>Real Simulator Captures</h2>
    <div class="grid-2">
      <div class="lumen-media"><img src="assets/launch/sensor-layer.png" alt="Lumen multimodal sensor layer"></div>
      <div class="lumen-media"><img src="assets/launch/physics-layer.png" alt="Lumen flow, clot, aneurysm, and device state"></div>
      <div class="lumen-media"><img src="assets/launch/nav-frame.png" alt="Lumen navigation benchmark rollouts"></div>
      <div class="lumen-media"><img src="assets/launch/benchmark-outro.png" alt="Lumen reproducible benchmark launch frame"></div>
    </div>
  </section>

  <section class="section">
    <h2>Why It Moves Beyond CathSim</h2>
    <p>
      CathSim made open endovascular RL research easier to start. Lumen is aimed at the next benchmark layer: deformable-wall semantics, wall-safety scoring, paired image/state observations, dataset-grade labels, and endovascular modules that expose flow, aneurysm, clot, and device effects. The result is a stronger public substrate for agents that must optimize more than reaching a coordinate.
    </p>
  </section>

  <section class="section">
    <h2>Run It</h2>
    <pre class="code-block"><code>git clone https://github.com/SeldingerMed/seldinger-lumen
cd seldinger-lumen
pip install -e ".[dev]"
lumen doctor
lumen play stenotic --out lumen-run
lumen benchmark lumen-bench
lumen capture lumen-episodes
lumen validate lumen-episodes --require-cv-labels</code></pre>
  </section>

  <section class="section">
    <h2>Research Package</h2>
    <ul>
      <li><a href="assets/launch/lumen-preprint.pdf">Read the launch preprint PDF</a></li>
      <li><a href="assets/launch/lumen-preprint-latex.zip">Download the LaTeX source ZIP</a></li>
      <li><a href="assets/launch/social-media-proposals.md">Open the launch post drafts</a></li>
      <li><a href="https://github.com/SeldingerMed/seldinger-lumen">Open the public repository</a></li>
    </ul>
  </section>
</div>
