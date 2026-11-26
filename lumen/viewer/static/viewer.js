/* Lumen viewer: a dependency-free WebGL2 page.
 *
 * Left pane: the vessels in 3D, see-through so the wire shows inside, free to rotate.
 * Right pane: a C-arm view. Each pixel's darkness comes from how much contrast-filled vessel
 * (and wire) the ray crosses: front faces subtract their depth, back faces add it, so the
 * sum is the path length through the lumen, and the image is exp(-mu * length) over a soft
 * tissue background, with quantum noise and the round field of an image intensifier.
 *
 * Coordinates are the anatomy's own millimetres (LPS for imported scans: x = patient left,
 * y = posterior, z = superior). Angles follow the C-arm convention: LAO is positive, RAO
 * negative; cranial positive, caudal negative.
 */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("lumen-data").textContent);
  const S = DATA.scene, R = DATA.replay;
  const $ = (id) => document.getElementById(id);
  const canvas = $("gl");
  const gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
  if (!gl) { $("nogl").style.display = "block"; return; }

  // ------------------------------------------------------------------ small vector maths
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const rad = (d) => d * Math.PI / 180;
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  }
  function lookAt(eye, at, up) {
    const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  }
  function mmul(a, b) {
    const o = new Array(16).fill(0);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return o;
  }
  // direction from the patient to the viewer (the image intensifier) for C-arm angles
  function viewDir(lao, cra) {
    const a = rad(lao), c = rad(cra);
    return norm([Math.sin(a) * Math.cos(c), -Math.cos(a) * Math.cos(c), Math.sin(c)]);
  }
  function upFor(dir) {
    let up = [0, 0, 1];
    if (Math.abs(dot(up, dir)) > 0.99) up = [0, -1, 0];
    return norm(sub(up, mul(dir, dot(up, dir))));
  }

  // ------------------------------------------------------------------ scene data
  function b64(s, T) {
    const bin = atob(s), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new T(u8.buffer);
  }
  const pos = b64(S.mesh.positions, Float32Array), idx = b64(S.mesh.indices, Uint32Array);
  const nrm = new Float32Array(pos.length);
  for (let t = 0; t < idx.length; t += 3) {
    const i = idx[t] * 3, j = idx[t + 1] * 3, k = idx[t + 2] * 3;
    const e1 = [pos[j] - pos[i], pos[j + 1] - pos[i + 1], pos[j + 2] - pos[i + 2]];
    const e2 = [pos[k] - pos[i], pos[k + 1] - pos[i + 1], pos[k + 2] - pos[i + 2]];
    const n = cross(e1, e2);
    for (const v of [i, j, k]) { nrm[v] += n[0]; nrm[v + 1] += n[1]; nrm[v + 2] += n[2]; }
  }
  for (let v = 0; v < nrm.length; v += 3) {
    const l = Math.hypot(nrm[v], nrm[v + 1], nrm[v + 2]) || 1;
    nrm[v] /= l; nrm[v + 1] /= l; nrm[v + 2] /= l;
  }
  const lo = S.bounds.min, hi = S.bounds.max;
  const center = mul(add(lo, hi), 0.5);
  const radius = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

  // ------------------------------------------------------------------ GL helpers
  function shader(vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, "#version 300 es\nprecision highp float;\n" + src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }
  function vao(arrays, index) {
    const v = gl.createVertexArray();
    gl.bindVertexArray(v);
    const bufs = [];
    arrays.forEach((a, loc) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, a, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      bufs.push(b);
    });
    let ib = null;
    if (index) {
      ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.DYNAMIC_DRAW);
    }
    gl.bindVertexArray(null);
    return { v, bufs, ib, count: index ? index.length : arrays[0].length / 3 };
  }
  function update(obj, arrays, index) {
    arrays.forEach((a, i) => { gl.bindBuffer(gl.ARRAY_BUFFER, obj.bufs[i]); gl.bufferData(gl.ARRAY_BUFFER, a, gl.DYNAMIC_DRAW); });
    if (index) { gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, obj.ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.DYNAMIC_DRAW); }
    obj.count = index ? index.length : arrays[0].length / 3;
  }

  const SURF = shader(`
    layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNrm;
    uniform mat4 uVP; out vec3 vN; out vec3 vW;
    void main(){ vN=aNrm; vW=aPos; gl_Position=uVP*vec4(aPos,1.); }`, `
    in vec3 vN; in vec3 vW; uniform vec3 uEye; uniform vec3 uKey; uniform vec3 uColor;
    uniform float uAlpha; uniform float uMetal; uniform vec3 uRimCol; uniform float uRim;
    uniform float uAmb; uniform float uSpec; uniform float uInner; out vec4 o;
    void main(){
      vec3 v=normalize(uEye-vW); vec3 n=normalize(vN); bool inside=dot(n,v)<0.; if(inside) n=-n;
      float ndl=max(dot(n,uKey),0.); float fill=max(dot(n,normalize(vec3(-.4,.3,-.2))),0.);
      float hemi=.5+.5*n.z; vec3 h=normalize(uKey+v);
      float spec=pow(max(dot(n,h),0.),mix(40.,90.,uMetal));
      float fres=pow(1.-max(dot(n,v),0.),3.);
      vec3 col=uColor*(uAmb+.66*ndl+.16*fill+.18*hemi)+spec*mix(uSpec,.9,uMetal)
               +uRimCol*fres*uRim*(1.-uMetal);
      float a=clamp(uAlpha+fres*(1.-uAlpha)*mix(.55,1.,step(.5,uRim)),0.,1.);
      if(inside){ col*=uInner; }
      o=vec4(col,a);
    }`);
  // three looks for the 3D pane; the C-arm pane is the same in all of them
  const LOOKS = {
    monitor: { bg: [0.055, 0.065, 0.085], vessel: [0.86, 0.24, 0.2], alpha: 0.6, rimCol: [1, 0.62, 0.55],
      rim: 0.28, amb: 0.26, spec: 0.35, inner: 0.45, wire: [0.78, 0.82, 0.88], through: [0.55, 0.9, 1.0],
      trail: [0.35, 0.85, 1.0, 0.55], mark: [0.3, 0.75, 1.0, 0.9] },
    anatomy: { bg: [0.93, 0.915, 0.89], vessel: [0.74, 0.13, 0.12], alpha: 1.0, rimCol: [0.35, 0.05, 0.05],
      rim: 0.0, amb: 0.2, spec: 0.22, inner: 0.6, wire: [0.2, 0.23, 0.27], through: [0.25, 0.8, 1.0],
      trail: [0.15, 0.6, 1.0, 0.85], mark: [0.1, 0.45, 0.95, 0.95] },
    glass: { bg: [0.008, 0.01, 0.018], vessel: [0.92, 0.86, 0.78], alpha: 0.16, rimCol: [1.0, 0.86, 0.7],
      rim: 0.95, amb: 0.12, spec: 0.5, inner: 0.35, wire: [0.5, 0.95, 1.0], through: [0.45, 0.95, 1.0],
      trail: [1.0, 0.75, 0.3, 0.6], mark: [1.0, 0.75, 0.3, 0.95] },
  };
  const FLAT = shader(`
    layout(location=0) in vec3 aPos; uniform mat4 uVP; uniform float uSize;
    void main(){ gl_Position=uVP*vec4(aPos,1.); gl_PointSize=uSize; }`, `
    uniform vec4 uColor; uniform float uRound; out vec4 o;
    void main(){ if(uRound>0.5){ vec2 d=gl_PointCoord-.5; if(dot(d,d)>.25) discard; } o=uColor; }`);
  // signed depth along the view axis: entering a surface subtracts, leaving adds
  const THICK = shader(`
    layout(location=0) in vec3 aPos; uniform mat4 uVP; uniform vec3 uDir; uniform vec3 uRef;
    out float vD; void main(){ vD=dot(uRef-aPos,uDir); gl_Position=uVP*vec4(aPos,1.); }`, `
    in float vD; uniform vec4 uMask; out vec4 o;
    void main(){ float s=gl_FrontFacing?-1.:1.; o=uMask*(s*vD); }`);
  const FLUORO = shader(`
    out vec2 uv; void main(){ vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); uv=p; gl_Position=vec4(p*2.-1.,0.,1.); }`, `
    in vec2 uv; uniform sampler2D uT; uniform float uContrast; uniform float uTime; uniform vec2 uRes;
    uniform float uScale; out vec4 o;
    float h(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
    float vn(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
      return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y); }
    float fbm(vec2 p){ float s=0.,a=.5; for(int k=0;k<5;k++){ s+=a*vn(p); p*=2.03; a*=.5; } return s; }
    void main(){
      vec4 t=texture(uT,uv);
      float tv=max(t.r,0.), tw=max(t.g,0.);
      vec2 q=(uv-.5)*vec2(uRes.x/uRes.y,1.);
      float body=.70+.10*fbm(q*2.2*uScale+3.1)-.10*smoothstep(.15,.65,length(q));
      float att=uContrast*.21*tv+9.0*tw;   // metal wire: far more opaque than iodine
      float I=body*exp(-att);
      I=mix(I,I*(.92+.16*fbm(q*40.)),.4);
      I+=(h(gl_FragCoord.xy+fract(uTime)*91.7)-.5)*.055;
      float r=length(q); float field=1.-smoothstep(.455,.47,r);
      I=I*field*(1.-.35*smoothstep(.25,.47,r));
      I=pow(clamp(I,0.,1.),.9);
      o=vec4(vec3(I*.97,I,I*1.02),1.);
    }`);

  const vessel = vao([pos, nrm], idx);
  const wireObj = vao([new Float32Array(3), new Float32Array(3)], new Uint32Array(3));
  const lineObj = vao([new Float32Array(3)]);
  const pointObj = vao([new Float32Array(3)]);

  // centrelines and markers
  const clPts = [];
  for (const c of S.centerlines) for (let i = 0; i + 1 < c.points.length; i++) clPts.push(...c.points[i], ...c.points[i + 1]);
  const centrelines = vao([new Float32Array(clPts)]);
  const route = R && R.route ? R.route : null;
  const routePts = [];
  if (route) for (let i = 0; i + 1 < route.length; i++) routePts.push(...route[i], ...route[i + 1]);
  const routeObj = vao([new Float32Array(routePts.length ? routePts : [0, 0, 0])]);
  const targetPos = route ? route[route.length - 1] : null;

  // ------------------------------------------------------------------ wire tube
  function catmull(pts, sub) {
    const out = [];
    for (let i = 0; i + 1 < pts.length; i++) {
      const p0 = pts[Math.max(i - 1, 0)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(i + 2, pts.length - 1)];
      for (let k = 0; k < sub; k++) {
        const t = k / sub, t2 = t * t, t3 = t2 * t;
        out.push([0, 1, 2].map((a) => 0.5 * ((2 * p1[a]) + (-p0[a] + p2[a]) * t +
          (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * t2 + (-p0[a] + 3 * p1[a] - 3 * p2[a] + p3[a]) * t3)));
      }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }
  function tube(nodes, r, sides) {
    const c = catmull(nodes, 5), n = c.length, P = [], N = [], I = [];
    let m = null;
    for (let i = 0; i < n; i++) {
      const t = norm(sub(c[Math.min(i + 1, n - 1)], c[Math.max(i - 1, 0)]));
      if (!m) { m = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; }
      m = norm(sub(m, mul(t, dot(m, t))));
      const b = cross(t, m);
      for (let j = 0; j < sides; j++) {
        const a = 2 * Math.PI * j / sides, d = add(mul(m, Math.cos(a)), mul(b, Math.sin(a)));
        P.push(...add(c[i], mul(d, r))); N.push(...d);
      }
    }
    for (let i = 0; i + 1 < n; i++) for (let j = 0; j < sides; j++) {
      const a = i * sides + j, b2 = i * sides + (j + 1) % sides;
      I.push(a, b2, a + sides, b2, b2 + sides, a + sides);
    }
    // close both ends so the C-arm pass sees a solid wire
    const c0 = P.length / 3; P.push(...c[0]); N.push(...norm(sub(c[0], c[1])));
    const c1 = P.length / 3; P.push(...c[n - 1]); N.push(...norm(sub(c[n - 1], c[n - 2])));
    for (let j = 0; j < sides; j++) {
      I.push(c0, (j + 1) % sides, j);
      I.push(c1, (n - 1) * sides + j, (n - 1) * sides + (j + 1) % sides);
    }
    return [new Float32Array(P), new Float32Array(N), new Uint32Array(I)];
  }

  // ------------------------------------------------------------------ state and UI
  const st = {
    lao: 0, cra: 0, dist: radius * 3.4, panX: 0, panY: 0,
    carm: { lao: 0, cra: 0, field: Math.min(2 * radius, 320) }, link: true,
    alpha: 0.6, labels: true, lines: false, contrast: true, layout: "both", look: "monitor",
    frame: 0, playing: false, speed: 1, acc: 0,
  };
  const frames = R ? R.frames : [];
  const trail = frames.map((f) => f.nodes[f.nodes.length - 1]);
  // drawn a little thicker than the device so it reads at any zoom, as on a monitor
  const wireR = R ? Math.max((R.device_radius_mm || 0.2) * 2, radius * 0.0045) : 0.3;

  function panes() {
    const w = canvas.width, h = canvas.height;
    if (st.layout === "3d") return { d3: [0, 0, w, h], carm: null };
    if (st.layout === "carm") return { d3: null, carm: [0, 0, w, h] };
    const half = Math.floor(w / 2);
    return { d3: [0, 0, half, h], carm: [half, 0, w - half, h] };
  }
  function cam3d(vp) {
    const dir = viewDir(st.lao, st.cra), up = upFor(dir), right = cross(dir, up);
    const at = add(center, add(mul(right, -st.panX), mul(up, st.panY)));
    const eye = add(at, mul(dir, st.dist));
    const P = perspective(rad(35), vp[2] / vp[3], st.dist * 0.05, st.dist + radius * 4);
    return { VP: mmul(P, lookAt(eye, at, up)), eye, dir };
  }
  function camCarm(vp) {
    const a = st.link ? { lao: st.lao, cra: st.cra } : st.carm;
    const dir = viewDir(a.lao, a.cra), up = upFor(dir), sod = 700;
    const eye = add(center, mul(dir, sod));
    const fov = 2 * Math.atan(st.carm.field / 2 / sod);
    const P = perspective(fov, vp[2] / vp[3], sod - radius * 2, sod + radius * 2);
    return { VP: mmul(P, lookAt(eye, center, up)), eye, dir };
  }

  // ------------------------------------------------------------------ fluoro target
  const floatOK = !!gl.getExtension("EXT_color_buffer_float");
  const blend32 = floatOK && !!gl.getExtension("EXT_float_blend");
  let fbo = null, tex = null, fboSize = [0, 0];
  function ensureFbo(w, h) {
    if (fbo && fboSize[0] === w && fboSize[1] === h) return;
    if (tex) gl.deleteTexture(tex);
    if (fbo) gl.deleteFramebuffer(fbo);
    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const fmt = blend32 ? gl.RGBA32F : (floatOK ? gl.RGBA16F : gl.RGBA8);
    const type = blend32 ? gl.FLOAT : (floatOK ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, gl.RGBA, type, null);
    const filt = blend32 ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fboSize = [w, h];
  }

  // ------------------------------------------------------------------ drawing
  function drawSurface(obj, cam, color, alpha, metal) {
    const L = LOOKS[st.look];
    gl.useProgram(SURF.p);
    gl.uniform3fv(SURF.u.uRimCol, L.rimCol); gl.uniform1f(SURF.u.uRim, L.rim);
    gl.uniform1f(SURF.u.uAmb, L.amb); gl.uniform1f(SURF.u.uSpec, L.spec); gl.uniform1f(SURF.u.uInner, L.inner);
    gl.uniformMatrix4fv(SURF.u.uVP, false, cam.VP);
    gl.uniform3fv(SURF.u.uEye, cam.eye);
    gl.uniform3fv(SURF.u.uKey, norm(add(cam.dir, add(upFor(cam.dir), mul(cross(cam.dir, upFor(cam.dir)), -0.6)))));
    gl.uniform3fv(SURF.u.uColor, color);
    gl.uniform1f(SURF.u.uAlpha, alpha);
    gl.uniform1f(SURF.u.uMetal, metal);
    gl.bindVertexArray(obj.v);
    gl.drawElements(gl.TRIANGLES, obj.count, gl.UNSIGNED_INT, 0);
  }
  function drawFlat(obj, cam, mode, color, size, round) {
    gl.useProgram(FLAT.p);
    gl.uniformMatrix4fv(FLAT.u.uVP, false, cam.VP);
    gl.uniform4fv(FLAT.u.uColor, color);
    gl.uniform1f(FLAT.u.uSize, size || 1);
    gl.uniform1f(FLAT.u.uRound, round ? 1 : 0);
    gl.bindVertexArray(obj.v);
    gl.drawArrays(mode, 0, obj.count);
  }
  function setWire() {
    if (!R || !frames.length) return false;
    const f = frames[st.frame];
    const [P, N, I] = tube(f.nodes, wireR, 10);
    update(wireObj, [P, N], I);
    return true;
  }

  function draw3d(vp, hasWire) {
    gl.viewport(...vp); gl.scissor(...vp);
    const L = LOOKS[st.look];
    gl.clearColor(...L.bg, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const cam = cam3d(vp);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.depthMask(true);
    if (hasWire) {
      const f = frames[st.frame];
      const bad = f.overlap_mm > 1.0;   // pressing into the wall
      drawSurface(wireObj, cam, bad ? [0.95, 0.25, 0.2] : L.wire, 1, 1);
      const upto = trail.slice(0, st.frame + 1);
      if (upto.length > 1) {
        const tp = []; for (let i = 0; i + 1 < upto.length; i++) tp.push(...upto[i], ...upto[i + 1]);
        update(lineObj, [new Float32Array(tp)]);
        drawFlat(lineObj, cam, gl.LINES, L.trail);
      }
    }
    if (st.lines) drawFlat(centrelines, cam, gl.LINES, [1, 0.85, 0.4, 0.6]);
    if (route) drawFlat(routeObj, cam, gl.LINES, [0.35, 1.0, 0.6, 0.35]);
    const marks = [S.entry].concat(targetPos ? [targetPos] : []);
    update(pointObj, [new Float32Array(marks.flat())]);
    gl.disable(gl.DEPTH_TEST);
    drawFlat(pointObj, cam, gl.POINTS, L.mark, 10 * devicePixelRatio, true);
    gl.enable(gl.DEPTH_TEST);
    // see-through vessel: inner walls first, then the near side
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(st.alpha >= 0.99);
    gl.enable(gl.CULL_FACE);
    if (st.look === "glass") gl.blendFunc(gl.SRC_ALPHA, gl.ONE);   // glow: walls add light
    gl.cullFace(gl.FRONT); drawSurface(vessel, cam, L.vessel, st.alpha, 0);
    gl.cullFace(gl.BACK); drawSurface(vessel, cam, L.vessel, st.alpha, 0);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE); gl.depthMask(true);
    if (hasWire) {   // the wire again, through the wall, so it is never lost inside the vessel
      const f = frames[st.frame];
      gl.disable(gl.DEPTH_TEST);
      if (st.alpha >= 0.99 && st.frame > 0) drawFlat(lineObj, cam, gl.LINES, L.trail);   // solid walls hide the path
      drawSurface(wireObj, cam, f.overlap_mm > 1.0 ? [1.0, 0.35, 0.3] : L.through, 0.9, 0.3);
      gl.enable(gl.DEPTH_TEST);
    }
    gl.disable(gl.BLEND);
    return cam;
  }

  function drawCarm(vp, hasWire, time) {
    ensureFbo(vp[2], vp[3]);
    const cam = camCarm(vp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, vp[2], vp[3]); gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.blendEquation(gl.FUNC_ADD);
    gl.useProgram(THICK.p);
    gl.uniformMatrix4fv(THICK.u.uVP, false, cam.VP);
    gl.uniform3fv(THICK.u.uDir, cam.dir);
    gl.uniform3fv(THICK.u.uRef, add(center, mul(cam.dir, radius * 1.5)));
    gl.uniform4fv(THICK.u.uMask, [1, 0, 0, 0]);
    gl.bindVertexArray(vessel.v); gl.drawElements(gl.TRIANGLES, vessel.count, gl.UNSIGNED_INT, 0);
    if (hasWire) {
      gl.uniform4fv(THICK.u.uMask, [0, 1, 0, 0]);
      gl.bindVertexArray(wireObj.v); gl.drawElements(gl.TRIANGLES, wireObj.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.SCISSOR_TEST);
    gl.viewport(...vp); gl.scissor(...vp);
    gl.disable(gl.BLEND);
    gl.useProgram(FLUORO.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(FLUORO.u.uT, 0);
    gl.uniform1f(FLUORO.u.uContrast, st.contrast ? 1 : 0.12);
    gl.uniform1f(FLUORO.u.uTime, time * 0.001);
    gl.uniform2fv(FLUORO.u.uRes, [vp[2], vp[3]]);
    gl.uniform1f(FLUORO.u.uScale, 300 / st.carm.field);
    gl.bindVertexArray(null);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return cam;
  }

  // labels: HTML tags placed over the 3D pane
  const labelBox = $("labels");
  const tags = S.labels.map((l) => { const d = document.createElement("div"); d.className = "tag"; d.textContent = l.name; labelBox.appendChild(d); return d; });
  function placeLabels(cam, vp) {
    const show = st.labels && cam;
    S.labels.forEach((l, i) => {
      const d = tags[i];
      if (!show) { d.style.display = "none"; return; }
      const p = l.at, M = cam.VP;
      const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13];
      const w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
      if (w <= 0) { d.style.display = "none"; return; }
      const dpr = devicePixelRatio;
      const sx = (vp[0] + (x / w * 0.5 + 0.5) * vp[2]) / dpr, sy = (canvas.height - (vp[1] + (y / w * 0.5 + 0.5) * vp[3])) / dpr;
      d.style.display = "block"; d.style.transform = `translate(${sx}px, ${sy}px)`;
    });
  }

  function hud() {
    const a = st.link ? { lao: st.lao, cra: st.cra } : st.carm;
    const fmt = (v, p, n) => (Math.abs(v) < 0.5 ? "0" : `${v > 0 ? p : n} ${Math.abs(Math.round(v))}`);
    $("angles").textContent = `${fmt(a.lao, "LAO", "RAO")}  ${fmt(a.cra, "CRA", "CAU")}`;
    $("lao").value = a.lao; $("cra").value = a.cra;
    if (!R) return;
    const f = frames[st.frame];
    $("scrub").value = st.frame;
    $("stepinfo").textContent = `step ${f.step} / ${frames[frames.length - 1].step}`;
    $("left").textContent = `${f.left_mm.toFixed(1)} mm to target`;
    const ov = $("overlap");
    ov.textContent = `wall overlap ${f.overlap_mm.toFixed(2)} mm`;
    ov.className = f.overlap_mm > 1.0 ? "warn" : "";
    $("play").textContent = st.playing ? "Pause" : "Play";
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (st.playing && frames.length) {
      st.acc += dt * 30 * st.speed;
      while (st.acc >= 1) { st.acc -= 1; st.frame++; if (st.frame >= frames.length) { st.frame = frames.length - 1; st.playing = false; break; } }
    }
    resize();
    const hasWire = setWire();
    const pn = panes();
    gl.enable(gl.SCISSOR_TEST);
    let cam = null;
    if (pn.d3) cam = draw3d(pn.d3, hasWire);
    if (pn.carm) drawCarm(pn.carm, hasWire, now);
    placeLabels(cam, pn.d3);
    hud();
    requestAnimationFrame(frame);
  }
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  // ------------------------------------------------------------------ input
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) { const k = st.dist / canvas.clientHeight; st.panX += dx * k; st.panY += dy * k; return; }
    st.lao = ((st.lao - dx * 0.4 + 540) % 360) - 180;
    st.cra = Math.max(-89, Math.min(89, st.cra + dy * 0.4));
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    st.dist = Math.max(radius * 0.3, Math.min(radius * 8, st.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
    const [l, c] = b.dataset.view.split(",").map(Number); st.lao = l; st.cra = c; st.panX = st.panY = 0;
  }));
  document.querySelectorAll("[data-layout]").forEach((b) => b.addEventListener("click", () => { st.layout = b.dataset.layout; }));
  $("alpha").addEventListener("input", (e) => { st.alpha = Number(e.target.value); });
  function setLook(name) {
    if (!LOOKS[name]) return;
    st.look = name; st.alpha = LOOKS[name].alpha;
    $("alpha").value = st.alpha; $("look").value = name;
    document.body.dataset.look = name;
  }
  $("look").addEventListener("change", (e) => setLook(e.target.value));
  setLook((location.hash.match(/look=(\w+)/) || [])[1] || S.look || "monitor");
  $("showlabels").addEventListener("change", (e) => { st.labels = e.target.checked; });
  $("showlines").addEventListener("change", (e) => { st.lines = e.target.checked; });
  $("contrast").addEventListener("change", (e) => { st.contrast = e.target.checked; });
  $("link").addEventListener("change", (e) => { st.link = e.target.checked; if (!st.link) { st.carm.lao = st.lao; st.carm.cra = st.cra; } });
  $("lao").addEventListener("input", (e) => { if (st.link) st.lao = Number(e.target.value); else st.carm.lao = Number(e.target.value); });
  $("cra").addEventListener("input", (e) => { if (st.link) st.cra = Number(e.target.value); else st.carm.cra = Number(e.target.value); });
  $("field").addEventListener("input", (e) => { st.carm.field = Number(e.target.value); });
  $("field").max = Math.max(400, Math.ceil(2 * radius)); $("field").value = st.carm.field;
  if (R) {
    $("replay").style.display = "flex";
    $("scrub").max = frames.length - 1;
    $("scrub").addEventListener("input", (e) => { st.frame = Number(e.target.value); st.playing = false; });
    $("play").addEventListener("click", () => { if (st.frame >= frames.length - 1) st.frame = 0; st.playing = !st.playing; });
    $("speed").addEventListener("change", (e) => { st.speed = Number(e.target.value); });
    const o = R.outcome || {};
    $("outcome").textContent = o.reached ? `reached ${R.target || "the target"}` : `stopped ${o.left_mm ?? "?"} mm short`;
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") { e.preventDefault(); $("play").click(); }
      if (e.code === "ArrowRight") st.frame = Math.min(frames.length - 1, st.frame + 1);
      if (e.code === "ArrowLeft") st.frame = Math.max(0, st.frame - 1);
    });
  }
  $("title").textContent = S.label;
  const src = S.source || {};
  $("meta").textContent = [S.mesh.kind, src.licence ? `${src.licence}` : "", src.citation || ""].filter(Boolean).join("  ·  ");
  $("meta").title = $("meta").textContent;
  if (!floatOK) $("meta").textContent += "  ·  C-arm view: this browser lacks float render targets";
  window.lumenViewer = { state: st, scene: S, replay: R, setLook, looks: Object.keys(LOOKS), ready: true };
  requestAnimationFrame(frame);
})();
