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
      <div class="lumen-kicker">Lumen</div>
      <h1>An open simulator for endovascular navigation research.</h1>
      <p class="lumen-lede">
        Lumen is an Apache-2.0 environment for catheter and guidewire navigation. It includes procedural vascular cases, safety-scored rollouts, synthetic fluoroscopy, luminal RGB, masks, keypoints, replay metadata, and Gymnasium tasks.
      </p>
      <div class="lumen-actions">
        <a class="lumen-button primary" href="https://github.com/SeldingerMed/seldinger-lumen">GitHub</a>
        <a class="lumen-button" href="assets/launch/lumen-preprint.pdf">Preprint PDF</a>
      </div>
    </div>
    <div class="lumen-media">
      <video
        src="assets/launch/lumen-launch.mp4"
        poster="assets/launch/physics-layer.png"
        controls
        muted
        playsinline
      ></video>
    </div>
  </section>

  <section class="section">
    <h2>What Is Included</h2>
    <div class="grid-3">
      <div class="metric"><strong>Safe target reach</strong> A run can reach the target and still be marked unsafe if wall or force limits are exceeded.</div>
      <div class="metric"><strong>Device route state</strong> Route progress, contact, penetration, torsion, and friction hooks are recorded during navigation.</div>
      <div class="metric"><strong>Image outputs</strong> Fluoroscopy, masks, keypoints, detector noise, and luminal RGB come from the same case state.</div>
      <div class="metric"><strong>Procedure modules</strong> Flow diversion, aneurysm inflow, clot fields, retrieval, and fragmentation are exposed as state.</div>
      <div class="metric"><strong>Replayable cases</strong> Episode sidecars, captures, indexes, and splits are built for reruns and outside inspection.</div>
      <div class="metric"><strong>Release files</strong> Code, benchmark summaries, preprint, screenshots, and launch materials are collected here.</div>
    </div>
  </section>

  <section class="section">
    <h2>Benchmark Result</h2>
    <p>
      In a matched branch-navigation PPO run, both environments trained for 50,000 steps and were evaluated for 30 deterministic held-out episodes. Lumen reached 100% raw success and 100% safe success on <code>nav_tree_branch</code>. CathSim reached 100% raw success on <code>phantom3_bca</code>, but 6.7% safe success under the comparison force threshold.
    </p>
    <div class="grid-3">
      <div class="metric"><strong>100%</strong> Lumen safe success across 30 PPO eval episodes.</div>
      <div class="metric"><strong>6.7%</strong> CathSim safe success under the matched force threshold.</div>
      <div class="metric"><strong>79.7 vs 12.1</strong> Eval steps/s for Lumen vs CathSim in the matched run.</div>
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
      <li><a href="assets/launch/benchmark/ppo-short-50k-lumen-cathsim-summary.csv">Download the matched PPO benchmark CSV</a></li>
      <li><a href="assets/launch/benchmark/pilot-summary-lumen-cathsim-steve.csv">Download the Lumen/CathSim/stEVE pilot CSV</a></li>
      <li><a href="assets/launch/social-media-proposals.md">Open the launch post drafts</a></li>
      <li><a href="https://github.com/SeldingerMed/seldinger-lumen">Open the public repository</a></li>
    </ul>
  </section>
</div>
