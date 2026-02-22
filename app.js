/* OAS OMR Marker (GitHub Pages, client-side)
   - Loads template + key + student PDFs
   - Aligns pages to template using ORB + homography
   - Builds bubble layout from template (Hough circles + clustering)
   - Reads student number grid + Q1-45
   - Outputs CSV (student_number, score) + optional annotated PDF
*/

/* global pdfjsLib, PDFLib, cv */

(() => {
  'use strict';

  // ---------- UI ----------
  const $ = (id) => document.getElementById(id);

  const fileTemplate = $('fileTemplate');
  const fileKey = $('fileKey');
  const fileStudents = $('fileStudents');

  const btnLoadTemplateFromManifest = $('btnLoadTemplateFromManifest');
  const btnLoadKeyFromManifest = $('btnLoadKeyFromManifest');
  const btnLoadStudentsFromManifest = $('btnLoadStudentsFromManifest');

  const chkRebuildLayout = $('chkRebuildLayout');
  const chkAnnotated = $('chkAnnotated');

  const btnRun = $('btnRun');
  const btnReset = $('btnReset');

  const bar = $('bar');
  const logEl = $('log');

  const btnDownloadCSV = $('btnDownloadCSV');
  const btnDownloadAnnotated = $('btnDownloadAnnotated');
  const previewTable = $('preview');

  function log(msg) {
    const t = new Date().toLocaleTimeString();
    logEl.textContent += `[${t}] ${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setProgress(p) {
    bar.style.width = `${Math.max(0, Math.min(100, p))}%`;
  }

  function resetUI() {
    logEl.textContent = '';
    setProgress(0);
    btnDownloadCSV.disabled = true;
    btnDownloadAnnotated.disabled = true;
    previewTable.innerHTML = '';
    state.outputs.csv = null;
    state.outputs.annotatedPdfBytes = null;
  }

  // ---------- Config ----------
  const CFG = {
    dpiScale: 2.0, // render scale (pdf.js). Increase for better accuracy, but slower.
    // Regions in normalized coords (x1,y1,x2,y2)
    p1_answers_region: [0.05, 0.70, 0.95, 0.97],
    p1_student_number_region: [0.40, 0.23, 0.86, 0.53],
    p2_first_column_region: [0.05, 0.12, 0.28, 0.58],

    options: 'ABCDE',
    q2_start: 41,
    q2_count: 5,

    // Hough circles
    hough_dp: 1.2,
    hough_min_dist: 18,
    hough_param1: 120,
    hough_param2: 22,
    hough_min_radius: 10,
    hough_max_radius: 26,

    // Fill thresholds
    answer_min_fill: 0.18,
    digit_min_fill: 0.12,
    multi_delta: 0.06
  };

  // ---------- State ----------
  const state = {
    manifest: null,
    template: { bytes: null, p1: null, p2: null, orb1: null, orb2: null },
    key: { bytes: null, answers: null },
    students: { files: [], bytes: [], results: [] },
    layout: null,
    outputs: { csv: null, annotatedPdfBytes: null },
    ready: { cv: false, pdf: false }
  };

  // ---------- Load libs readiness ----------
  // pdf.js worker
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.js';

  // Wait for OpenCV runtime
  const cvReady = new Promise((resolve) => {
    const check = () => {
      if (typeof cv !== 'undefined' && cv && cv.Mat) {
        if (cv['onRuntimeInitialized']) {
          cv['onRuntimeInitialized'] = () => resolve();
        } else {
          resolve();
        }
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });

  // ---------- Helpers ----------
  async function fetchBytes(path) {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Failed to fetch: ${path}`);
    return new Uint8Array(await res.arrayBuffer());
  }

  function fileToBytes(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(new Uint8Array(r.result));
      r.onerror = () => reject(new Error('File read failed'));
      r.readAsArrayBuffer(file);
    });
  }

  async function loadManifest() {
    const bytes = await fetchBytes('data/manifest.json');
    const txt = new TextDecoder().decode(bytes);
    state.manifest = JSON.parse(txt);
    return state.manifest;
  }

  function relToAbs(region, w, h) {
    const [x1,y1,x2,y2] = region;
    return [Math.floor(x1*w), Math.floor(y1*h), Math.floor(x2*w), Math.floor(y2*h)];
  }

  function median(values) {
    if (!values.length) return NaN;
    const a = values.slice().sort((x,y)=>x-y);
    const mid = Math.floor(a.length/2);
    return a.length % 2 ? a[mid] : (a[mid-1] + a[mid]) / 2;
  }

  // ---------- PDF rendering ----------
  async function renderPdfPageToCanvas(pdfBytes, pageIndex, scale) {
    const loadingTask = pdfjsLib.getDocument({ data: pdfBytes });
    const pdf = await loadingTask.promise;
    const page = await pdf.getPage(pageIndex + 1);
    const viewport = page.getViewport({ scale });

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = Math.floor(viewport.width);
    canvas.height = Math.floor(viewport.height);

    await page.render({ canvasContext: ctx, viewport }).promise;
    return canvas;
  }

  function canvasToMat(canvas) {
    return cv.imread(canvas); // RGBA
  }

  function matToCanvas(mat) {
    const canvas = document.createElement('canvas');
    canvas.width = mat.cols;
    canvas.height = mat.rows;
    cv.imshow(canvas, mat);
    return canvas;
  }

  // ---------- ORB Alignment ----------
  function buildOrbTemplate(templateRGBA) {
    const gray = new cv.Mat();
    cv.cvtColor(templateRGBA, gray, cv.COLOR_RGBA2GRAY);

    const orb = new cv.ORB(4000);
    const kp = new cv.KeyPointVector();
    const des = new cv.Mat();
    orb.detectAndCompute(gray, new cv.Mat(), kp, des);

    gray.delete();
    return { orb, kp, des };
  }

  function alignToTemplate(scanRGBA, templateRGBA, templateOrb) {
    const scanGray = new cv.Mat();
    cv.cvtColor(scanRGBA, scanGray, cv.COLOR_RGBA2GRAY);

    const orb = new cv.ORB(4000);
    const kpS = new cv.KeyPointVector();
    const desS = new cv.Mat();
    orb.detectAndCompute(scanGray, new cv.Mat(), kpS, desS);

    if (desS.empty() || templateOrb.des.empty() || kpS.size() < 20) {
      scanGray.delete(); kpS.delete(); desS.delete(); orb.delete();
      throw new Error('Alignment failed: not enough features');
    }

    const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(desS, templateOrb.des, knn, 2);

    // ratio test
    const good = [];
    for (let i = 0; i < knn.size(); i++) {
      const m = knn.get(i).get(0);
      const n = knn.get(i).get(1);
      if (m.distance < 0.75 * n.distance) good.push(m);
    }

    if (good.length < 25) {
      scanGray.delete(); kpS.delete(); desS.delete(); orb.delete(); bf.delete(); knn.delete();
      throw new Error('Alignment failed: insufficient good matches');
    }

    // Build point arrays
    const srcPts = [];
    const dstPts = [];
    for (const m of good) {
      const pS = kpS.get(m.queryIdx).pt;
      const pT = templateOrb.kp.get(m.trainIdx).pt;
      srcPts.push(pS.x, pS.y);
      dstPts.push(pT.x, pT.y);
    }

    const srcMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, srcPts);
    const dstMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, dstPts);
    const mask = new cv.Mat();
    const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask);

    if (H.empty()) {
      scanGray.delete(); kpS.delete(); desS.delete(); orb.delete(); bf.delete(); knn.delete();
      srcMat.delete(); dstMat.delete(); mask.delete(); H.delete();
      throw new Error('Alignment failed: homography not found');
    }

    const aligned = new cv.Mat();
    const dsize = new cv.Size(templateRGBA.cols, templateRGBA.rows);
    cv.warpPerspective(scanRGBA, aligned, H, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

    // cleanup
    scanGray.delete(); kpS.delete(); desS.delete(); orb.delete(); bf.delete(); knn.delete();
    srcMat.delete(); dstMat.delete(); mask.delete(); H.delete();

    return aligned;
  }

  // ---------- Circle detection & layout building ----------
  function detectCirclesInRegion(rgba, region) {
    const [x1,y1,x2,y2] = relToAbs(region, rgba.cols, rgba.rows);
    const roi = rgba.roi(new cv.Rect(x1, y1, x2-x1, y2-y1));

    const gray = new cv.Mat();
    cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
    cv.medianBlur(gray, gray, 5);

    const circles = new cv.Mat();
    cv.HoughCircles(
      gray, circles, cv.HOUGH_GRADIENT,
      CFG.hough_dp, CFG.hough_min_dist,
      CFG.hough_param1, CFG.hough_param2,
      CFG.hough_min_radius, CFG.hough_max_radius
    );

    const out = [];
    for (let i = 0; i < circles.cols; i++) {
      const cx = circles.data32F[i*3 + 0] + x1;
      const cy = circles.data32F[i*3 + 1] + y1;
      const r  = circles.data32F[i*3 + 2];
      out.push([cx, cy, r]);
    }

    roi.delete(); gray.delete(); circles.delete();
    return out;
  }

  function clusterByGap(vals, gap) {
    const a = vals.slice().sort((x,y)=>x-y);
    const clusters = [];
    let cur = [a[0]];
    for (let i=1;i<a.length;i++){
      const v = a[i];
      if (v - cur[cur.length-1] <= gap) cur.push(v);
      else { clusters.push(cur); cur = [v]; }
    }
    clusters.push(cur);
    const centers = clusters.map(c => c.reduce((s,v)=>s+v,0)/c.length);
    return { centers, clusters };
  }

  function buildLayoutFromTemplate(t1AlignedRGBA, t2AlignedRGBA) {
    // Page 1 circles in answers region
    const cAns = detectCirclesInRegion(t1AlignedRGBA, CFG.p1_answers_region);
    if (cAns.length < 150) throw new Error('Layout build failed: too few circles in page-1 answers region');

    const ys = cAns.map(c => c[1]);
    let centersY = null;
    let gap = 8;
    for (let k=0;k<12;k++){
      const {centers} = clusterByGap(ys, gap);
      if (centers.length <= 10) { centersY = centers.slice().sort((a,b)=>a-b).slice(0,10); break; }
      gap += 3;
    }
    if (!centersY || centersY.length < 10) {
      // fallback: 10 quantiles
      const sorted = ys.slice().sort((a,b)=>a-b);
      centersY = [];
      for (let i=0;i<10;i++){
        const q = (i+0.5)/10;
        centersY.push(sorted[Math.floor(q*(sorted.length-1))]);
      }
      centersY.sort((a,b)=>a-b);
    }

    const questions = {};
    const dy = 18;
    const opts = CFG.options.split('');

    for (let row_i=0; row_i<centersY.length; row_i++){
      const y0 = centersY[row_i];
      const rowCircles = cAns.filter(c => Math.abs(c[1]-y0) <= dy).sort((a,b)=>a[0]-b[0]);
      if (rowCircles.length < 10) continue;

      // group by big gaps into 4 blocks
      const groups = [];
      let cur = [rowCircles[0]];
      for (let i=1;i<rowCircles.length;i++){
        const c = rowCircles[i];
        if (c[0] - cur[cur.length-1][0] > 70) { groups.push(cur); cur=[c]; }
        else cur.push(c);
      }
      groups.push(cur);

      // take 4 largest groups
      groups.sort((a,b)=>b.length-a.length);
      const top4 = groups.slice(0,4);
      top4.sort((a,b)=> (a.reduce((s,x)=>s+x[0],0)/a.length) - (b.reduce((s,x)=>s+x[0],0)/b.length));

      for (let block_i=0; block_i<top4.length; block_i++){
        const g = top4[block_i].slice().sort((a,b)=>a[0]-b[0]).slice(0, opts.length);
        if (g.length !== opts.length) continue;

        const qnum = block_i*10 + (row_i+1); // 1..40
        questions[String(qnum)] = opts.map((opt, j) => ({ opt, x: g[j][0], y: g[j][1], r: g[j][2] }));
      }
    }

    // Student number circles
    const cSN = detectCirclesInRegion(t1AlignedRGBA, CFG.p1_student_number_region);
    if (cSN.length < 60) throw new Error('Layout build failed: too few circles in student-number region');
    cSN.sort((a,b)=>a[0]-b[0]);

    const xs = cSN.map(c=>c[0]);
    let colCenters = null;
    let colGap = 12;
    for (let k=0;k<12;k++){
      const {centers} = clusterByGap(xs, colGap);
      if (centers.length >= 6 && centers.length <= 12) { colCenters = centers.slice().sort((a,b)=>a-b); break; }
      colGap += 3;
    }
    if (!colCenters) colCenters = clusterByGap(xs, 25).centers.slice().sort((a,b)=>a-b);

    const columns = [];
    for (const x0 of colCenters){
      const col = cSN.filter(c=>Math.abs(c[0]-x0) <= 14).sort((a,b)=>a[1]-b[1]);
      if (col.length < 8) continue;
      const ysCol = col.map(c=>c[1]);
      const yCenters = clusterByGap(ysCol, 10).centers.slice().sort((a,b)=>a-b).slice(0,10);

      const digitCircles = yCenters.map(yc => col.reduce((best,c)=> (Math.abs(c[1]-yc) < Math.abs(best[1]-yc) ? c : best), col[0]));
      digitCircles.sort((a,b)=>a[1]-b[1]);

      const digits = [];
      for (let i=0;i<Math.min(10, digitCircles.length); i++){
        digits.push({ digit: i, x: digitCircles[i][0], y: digitCircles[i][1], r: digitCircles[i][2] });
      }
      columns.push(digits);
    }

    // Page 2: first column, top 5 rows => Q41-45
    const cP2 = detectCirclesInRegion(t2AlignedRGBA, CFG.p2_first_column_region);
    if (cP2.length < 40) throw new Error('Layout build failed: too few circles in page-2 first-column region');

    const ys2 = cP2.map(c=>c[1]).slice().sort((a,b)=>a-b);
    const yCenters2 = clusterByGap(ys2, 9).centers.slice().sort((a,b)=>a-b).slice(0, CFG.q2_count);

    for (let i=0;i<yCenters2.length;i++){
      const y0 = yCenters2[i];
      const row = cP2.filter(c=>Math.abs(c[1]-y0) <= 14).sort((a,b)=>a[0]-b[0]).slice(0, opts.length);
      if (row.length !== opts.length) continue;
      const qnum = CFG.q2_start + i;
      questions[String(qnum)] = opts.map((opt, j) => ({ opt, x: row[j][0], y: row[j][1], r: row[j][2] }));
    }

    if (!questions["1"] || !questions[String(CFG.q2_start + CFG.q2_count - 1)]) {
      throw new Error('Layout build failed: missing required questions. Adjust regions if needed.');
    }

    return { questions, studentNumber: { columns } };
  }

  // ---------- Fill scoring ----------
  function fillRatio(gray, cx, cy, r) {
    const cxI = Math.round(cx), cyI = Math.round(cy);
    const rIn = Math.max(3, Math.round(r * 0.55));
    const x1 = Math.max(0, cxI - rIn), x2 = Math.min(gray.cols, cxI + rIn);
    const y1 = Math.max(0, cyI - rIn), y2 = Math.min(gray.rows, cyI + rIn);
    const roi = gray.roi(new cv.Rect(x1, y1, x2-x1, y2-y1));
    if (roi.rows === 0 || roi.cols === 0) { roi.delete(); return 0.0; }

    const bw = new cv.Mat();
    cv.threshold(roi, bw, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

    const mask = new cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8UC1);
    cv.circle(mask, new cv.Point(Math.floor(roi.cols/2), Math.floor(roi.rows/2)), Math.floor(rIn*0.85), new cv.Scalar(255), -1);

    const masked = new cv.Mat();
    cv.bitwise_and(bw, mask, masked);

    const filled = cv.countNonZero(masked);
    const total = cv.countNonZero(mask);

    roi.delete(); bw.delete(); mask.delete(); masked.delete();

    return total ? (filled / total) : 0.0;
  }

  function pick(scores, minFill, multiDelta) {
    const items = Object.entries(scores).sort((a,b)=>b[1]-a[1]);
    const [bestOpt, best] = items[0];
    const second = items.length > 1 ? items[1][1] : 0;

    if (best < minFill) return { choice: null, status: 'blank' };
    if ((best - second) < multiDelta) return { choice: null, status: 'multi' };
    return { choice: bestOpt, status: 'ok' };
  }

  function readAnswers(alignedRGBA, layout, qnums, minFill) {
    const gray = new cv.Mat();
    cv.cvtColor(alignedRGBA, gray, cv.COLOR_RGBA2GRAY);

    const answers = {};
    const status = {};
    for (const q of qnums) {
      const bubbles = layout.questions[String(q)];
      const scores = {};
      for (const b of bubbles) scores[b.opt] = fillRatio(gray, b.x, b.y, b.r);
      const {choice, status: st} = pick(scores, minFill, CFG.multi_delta);
      answers[q] = choice;
      status[q] = st;
    }
    gray.delete();
    return { answers, status };
  }

  function readStudentNumber(alignedRGBA, layout) {
    const gray = new cv.Mat();
    cv.cvtColor(alignedRGBA, gray, cv.COLOR_RGBA2GRAY);

    const digits = [];
    for (const col of layout.studentNumber.columns) {
      const scores = {};
      for (const d of col) scores[String(d.digit)] = fillRatio(gray, d.x, d.y, d.r);
      const {choice} = pick(scores, CFG.digit_min_fill, CFG.multi_delta);
      digits.push(choice ?? '');
    }
    gray.delete();
    const sn = digits.join('').trim();
    return sn || 'UNKNOWN';
  }

  function annotatePage(alignedRGBA, layout, answers, key, qnums) {
    const img = alignedRGBA.clone();

    for (const q of qnums) {
      const bubbles = layout.questions[String(q)];
      // light rings
      for (const b of bubbles) cv.circle(img, new cv.Point(b.x, b.y), Math.round(b.r), new cv.Scalar(180,180,180,255), 1);

      const chosen = answers[q];
      if (!chosen) continue;
      const bsel = bubbles.find(b => b.opt === chosen);
      if (!bsel) continue;
      const correct = chosen === key[q];
      const color = correct ? new cv.Scalar(0,200,0,255) : new cv.Scalar(0,0,255,255);
      cv.circle(img, new cv.Point(bsel.x, bsel.y), Math.round(bsel.r*1.2), color, 3);
    }
    return img;
  }

  // ---------- CSV / downloads ----------
  function toCSV(rows) {
    const lines = ['student_number,score'];
    for (const r of rows) lines.push(`${r.student_number},${r.score}`);
    return lines.join('\n');
  }

  function downloadBytes(filename, bytes, mime) {
    const blob = new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function downloadText(filename, text, mime) {
    downloadBytes(filename, new TextEncoder().encode(text), mime);
  }

  function renderPreview(rows) {
    if (!rows.length) { previewTable.innerHTML = '<tr><td>No data</td></tr>'; return; }
    const head = '<tr><th>student_number</th><th>score</th></tr>';
    const body = rows.slice(0,10).map(r => `<tr><td>${r.student_number}</td><td>${r.score}</td></tr>`).join('');
    previewTable.innerHTML = head + body;
  }

  // ---------- Main workflow ----------
  async function ensureReady() {
    log('Loading libraries...');
    await cvReady;
    state.ready.cv = true;
    log('OpenCV.js ready.');
    state.ready.pdf = true;
    log('pdf.js ready.');
  }

  async function loadTemplate(bytes) {
    log('Rendering template PDF...');
    const c1 = await renderPdfPageToCanvas(bytes, 0, CFG.dpiScale);
    const c2 = await renderPdfPageToCanvas(bytes, 1, CFG.dpiScale);

    const m1 = canvasToMat(c1);
    const m2 = canvasToMat(c2);

    state.template.bytes = bytes;
    state.template.p1 = m1;
    state.template.p2 = m2;

    log(`Template rendered: page1 ${m1.cols}x${m1.rows}, page2 ${m2.cols}x${m2.rows}.`);

    // build ORB templates
    log('Preparing ORB template features...');
    state.template.orb1 = buildOrbTemplate(m1);
    state.template.orb2 = buildOrbTemplate(m2);
    log('ORB templates ready.');
  }

  async function buildLayoutIfNeeded() {
    if (!chkRebuildLayout.checked && state.layout) return;
    log('Building bubble layout from template...');
    // Template is already "aligned to itself"
    state.layout = buildLayoutFromTemplate(state.template.p1, state.template.p2);
    log(`Layout built. Questions mapped: ${Object.keys(state.layout.questions).length}. Student-number columns: ${state.layout.studentNumber.columns.length}.`);
  }

  async function readKey(bytes) {
    log('Rendering teacher key...');
    const c1 = await renderPdfPageToCanvas(bytes, 0, CFG.dpiScale);
    const c2 = await renderPdfPageToCanvas(bytes, 1, CFG.dpiScale);
    const m1 = canvasToMat(c1);
    const m2 = canvasToMat(c2);

    // Align to template
    log('Aligning teacher key to template...');
    const a1 = alignToTemplate(m1, state.template.p1, state.template.orb1);
    const a2 = alignToTemplate(m2, state.template.p2, state.template.orb2);

    const qnums = [...Array(40)].map((_,i)=>i+1).concat([...Array(CFG.q2_count)].map((_,i)=>CFG.q2_start+i));
    const {answers: key1} = readAnswers(a1, state.layout, qnums.filter(q=>q<=40), CFG.answer_min_fill);
    const {answers: key2} = readAnswers(a2, state.layout, qnums.filter(q=>q>=CFG.q2_start), CFG.answer_min_fill);

    const key = {};
    for (const q of qnums) key[q] = (q<=40 ? key1[q] : key2[q]);
    const bad = qnums.filter(q=>!key[q]);
    if (bad.length) throw new Error(`Teacher key has blank/ambiguous answers at: ${bad.join(', ')}`);

    // cleanup
    m1.delete(); m2.delete(); a1.delete(); a2.delete();

    state.key.bytes = bytes;
    state.key.answers = key;
    log('Teacher key read successfully.');
  }

  async function loadStudentsFromFiles(fileList) {
    state.students.files = Array.from(fileList);
    state.students.bytes = [];
    for (const f of state.students.files) state.students.bytes.push(await fileToBytes(f));
    log(`Loaded ${state.students.files.length} student PDFs (upload).`);
  }

  async function loadStudentsFromManifest() {
    const mf = state.manifest ?? await loadManifest();
    if (!mf.students || !mf.students.length) throw new Error('manifest.json has no students list.');
    const bytesList = [];
    for (const p of mf.students) bytesList.push(await fetchBytes(p));
    state.students.files = mf.students.map(p => ({ name: p.split('/').slice(-1)[0], fromManifest: true, path: p }));
    state.students.bytes = bytesList;
    log(`Loaded ${state.students.files.length} student PDFs (manifest).`);
  }

  async function markOneStudent(bytes, fileLabel) {
    const c1 = await renderPdfPageToCanvas(bytes, 0, CFG.dpiScale);
    const c2 = await renderPdfPageToCanvas(bytes, 1, CFG.dpiScale);

    const m1 = canvasToMat(c1);
    const m2 = canvasToMat(c2);

    const a1 = alignToTemplate(m1, state.template.p1, state.template.orb1);
    const a2 = alignToTemplate(m2, state.template.p2, state.template.orb2);

    const studentNumber = readStudentNumber(a1, state.layout);

    const qnums = [...Array(40)].map((_,i)=>i+1).concat([...Array(CFG.q2_count)].map((_,i)=>CFG.q2_start+i));
    const {answers: ans1} = readAnswers(a1, state.layout, qnums.filter(q=>q<=40), CFG.answer_min_fill);
    const {answers: ans2} = readAnswers(a2, state.layout, qnums.filter(q=>q>=CFG.q2_start), CFG.answer_min_fill);

    const answers = {};
    for (const q of qnums) answers[q] = (q<=40 ? ans1[q] : ans2[q]);

    let score = 0;
    for (const q of qnums) if (answers[q] && answers[q] === state.key.answers[q]) score++;

    // Annotate
    let ann1 = null, ann2 = null;
    if (chkAnnotated.checked) {
      ann1 = annotatePage(a1, state.layout, answers, state.key.answers, qnums.filter(q=>q<=40));
      ann2 = annotatePage(a2, state.layout, answers, state.key.answers, qnums.filter(q=>q>=CFG.q2_start));
    }

    // cleanup
    m1.delete(); m2.delete(); a1.delete(); a2.delete();
    if (ann1) ann1 = ann1; // keep
    if (ann2) ann2 = ann2;

    return { student_number: studentNumber, score, fileLabel, ann1, ann2 };
  }

  async function buildAnnotatedPdf(results) {
    const { PDFDocument } = PDFLib;
    const pdfDoc = await PDFDocument.create();

    for (const r of results) {
      for (const mat of [r.ann1, r.ann2]) {
        if (!mat) continue;
        const canvas = matToCanvas(mat);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
        const jpgBytes = Uint8Array.from(atob(dataUrl.split(',')[1]), c => c.charCodeAt(0));
        const jpg = await pdfDoc.embedJpg(jpgBytes);
        const page = pdfDoc.addPage([jpg.width, jpg.height]);
        page.drawImage(jpg, { x: 0, y: 0, width: jpg.width, height: jpg.height });
      }
    }

    const pdfBytes = await pdfDoc.save();
    return new Uint8Array(pdfBytes);
  }

  async function runMarking() {
    resetUI();
    btnRun.disabled = true;
    try {
      await ensureReady();

      // Ensure template and key and students are available
      if (!state.template.bytes) throw new Error('Template not loaded.');
      if (!state.key.bytes) throw new Error('Teacher key not loaded.');
      if (!state.students.bytes.length) throw new Error('No student PDFs loaded.');

      await buildLayoutIfNeeded();

      log('Marking students...');
      const results = [];
      const total = state.students.bytes.length;

      for (let i=0;i<total;i++){
        setProgress((i/total)*100);
        const label = state.students.files[i].name || state.students.files[i].path || `student_${i+1}.pdf`;
        log(`Processing ${i+1}/${total}: ${label}`);
        const r = await markOneStudent(state.students.bytes[i], label);
        results.push(r);
      }
      setProgress(100);

      // Build CSV rows with exactly 2 columns
      const rows = results.map(r => ({ student_number: r.student_number, score: r.score }));
      const csv = toCSV(rows);
      state.outputs.csv = csv;
      btnDownloadCSV.disabled = false;
      renderPreview(rows);

      if (chkAnnotated.checked) {
        log('Building annotated PDF...');
        const annBytes = await buildAnnotatedPdf(results);
        state.outputs.annotatedPdfBytes = annBytes;
        btnDownloadAnnotated.disabled = false;
        log('Annotated PDF ready.');
      }

      // release annotation mats
      for (const r of results) {
        if (r.ann1) r.ann1.delete();
        if (r.ann2) r.ann2.delete();
      }

      log('Done.');
    } catch (e) {
      log(`ERROR: ${e.message || e}`);
      console.error(e);
    } finally {
      btnRun.disabled = false;
    }
  }

  // ---------- Wiring ----------
  function maybeEnableRun() {
    btnRun.disabled = !(state.template.bytes && state.key.bytes && state.students.bytes.length);
  }

  btnReset.addEventListener('click', () => {
    // keep loaded libs; clear inputs + state
    resetUI();

    // Free mats
    if (state.template.p1) { state.template.p1.delete(); state.template.p1 = null; }
    if (state.template.p2) { state.template.p2.delete(); state.template.p2 = null; }
    state.template.bytes = null;
    state.template.orb1 = null;
    state.template.orb2 = null;

    state.key.bytes = null;
    state.key.answers = null;

    state.students.files = [];
    state.students.bytes = [];

    state.layout = null;

    fileTemplate.value = '';
    fileKey.value = '';
    fileStudents.value = '';

    maybeEnableRun();
    log('Reset.');
  });

  btnRun.addEventListener('click', runMarking);

  btnDownloadCSV.addEventListener('click', () => {
    if (!state.outputs.csv) return;
    downloadText('results.csv', state.outputs.csv, 'text/csv;charset=utf-8');
  });

  btnDownloadAnnotated.addEventListener('click', () => {
    if (!state.outputs.annotatedPdfBytes) return;
    downloadBytes('annotated.pdf', state.outputs.annotatedPdfBytes, 'application/pdf');
  });

  fileTemplate.addEventListener('change', async (e) => {
    resetUI();
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      const bytes = await fileToBytes(f);
      await ensureReady();
      // free old mats
      if (state.template.p1) state.template.p1.delete();
      if (state.template.p2) state.template.p2.delete();
      await loadTemplate(bytes);
      state.layout = null; // force rebuild
      maybeEnableRun();
      log('Template loaded (upload).');
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  fileKey.addEventListener('change', async (e) => {
    resetUI();
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      const bytes = await fileToBytes(f);
      await ensureReady();
      if (!state.template.bytes) throw new Error('Load template first.');
      await buildLayoutIfNeeded();
      await readKey(bytes);
      maybeEnableRun();
      log('Teacher key loaded (upload).');
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  fileStudents.addEventListener('change', async (e) => {
    resetUI();
    const fl = e.target.files;
    if (!fl || !fl.length) return;
    try {
      await loadStudentsFromFiles(fl);
      maybeEnableRun();
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  btnLoadTemplateFromManifest.addEventListener('click', async () => {
    resetUI();
    try {
      const mf = state.manifest ?? await loadManifest();
      const bytes = await fetchBytes(mf.template || 'data/template.pdf');
      await ensureReady();
      if (state.template.p1) state.template.p1.delete();
      if (state.template.p2) state.template.p2.delete();
      await loadTemplate(bytes);
      state.layout = null;
      maybeEnableRun();
      log('Template loaded (manifest).');
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  btnLoadKeyFromManifest.addEventListener('click', async () => {
    resetUI();
    try {
      const mf = state.manifest ?? await loadManifest();
      const keyPath = mf.key || 'data/key.pdf';
      const bytes = await fetchBytes(keyPath);
      await ensureReady();
      if (!state.template.bytes) throw new Error('Load template first.');
      await buildLayoutIfNeeded();
      await readKey(bytes);
      maybeEnableRun();
      log('Teacher key loaded (manifest).');
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  btnLoadStudentsFromManifest.addEventListener('click', async () => {
    resetUI();
    try {
      await loadStudentsFromManifest();
      maybeEnableRun();
    } catch (err) {
      log(`ERROR: ${err.message || err}`);
    }
  });

  // Initial
  (async () => {
    resetUI();
    log('Tip: Use manifest workflow for batching on GitHub Pages.');
    // try load manifest silently
    try { await loadManifest(); log('manifest.json loaded.'); } catch (_) {}
    maybeEnableRun();
  })();
})();
