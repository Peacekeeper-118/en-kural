/* En Kural – single-page UI (vanilla JS, no build step). */
(() => {
  'use strict';

  // ------------------------------------------------------------ auth token
  const params = new URLSearchParams(location.search);
  if (params.get('token')) {
    try { sessionStorage.setItem('enkural_token', params.get('token')); } catch (e) { /* ignore */ }
    history.replaceState({}, '', '/');
  }
  let TOKEN = '';
  try { TOKEN = sessionStorage.getItem('enkural_token') || ''; } catch (e) { TOKEN = ''; }

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const fmtTime = (s) => { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
  const fmtDur = (s) => { s = Math.round(s || 0); return s < 60 ? `${s}-second` : `${fmtTime(s)} minute`; };
  const fmtBytes = (b) => b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b > 1e6 ? `${Math.round(b / 1e6)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`;
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  let toastTimer = null;
  function toast(msg, isError) {
    const t = $('toast');
    t.textContent = msg; t.classList.toggle('error', !!isError); t.classList.remove('hidden');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), isError ? 8000 : 3500);
  }

  async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (TOKEN) headers['Authorization'] = `Bearer ${TOKEN}`;
    if (opts.json !== undefined) { headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(opts.json); }
    const res = await fetch(`/api${path}`, Object.assign({}, opts, { headers }));
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail || JSON.stringify(j); } catch (e) { /* ignore */ }
      if (res.status === 401) detail = 'Open the app with the link printed in the terminal (it contains your access key).';
      throw new Error(detail);
    }
    if (opts.blob) return res.blob();
    return res.json();
  }

  function upload(path, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api${path}`);
      if (TOKEN) xhr.setRequestHeader('Authorization', `Bearer ${TOKEN}`);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      xhr.onload = () => {
        let body = null; try { body = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error((body && body.detail) || `Upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error('The upload was interrupted. Check that the app is still running and try again.'));
      xhr.send(formData);
    });
  }

  // ------------------------------------------------------------ state
  const state = {
    system: null, profiles: [], profileId: null, sources: [], sourceId: null,
    job: null, buildJob: null, poll: null, buildPoll: null, result: null, ftJob: null, ftPoll: null,
    recBlob: null, blobUrls: [],
  };
  const busy = (j) => !!(j && ['queued', 'running'].includes(j.status));

  // ------------------------------------------------------------ quality tags
  // for recordings of you, a short file is fine: only the total amount of speech matters
  const issuesOf = (q, voice) => (q && q.issues || []).filter(i => !(voice && i.code === 'too_little_speech'));
  function qualityTag(q, voice) {
    if (!q) return el('span', 'tag info', 'Not checked');
    const issues = issuesOf(q, voice);
    const poor = issues.some(i => i.severity === 'poor');
    const warn = issues.length > 0;
    const t = el('span', `tag ${poor ? 'poor' : warn ? 'usable' : 'good'}`, poor ? 'Poor quality' : warn ? 'Usable' : 'Good quality');
    t.title = `Speech ${fmtTime(q.speech_duration)}, score ${q.score}/100`;
    return t;
  }
  function issueList(q, voice) {
    const ul = el('ul', 'issues');
    issuesOf(q, voice).forEach(i => ul.append(el('li', null, `${i.title}. ${i.detail}`)));
    return ul;
  }

  // ------------------------------------------------------------ progress panel
  const STAGES = [['upload', 'Starting'], ['analyze', 'Loading your voice'], ['extract', 'Preparing the recording'],
    ['convert', 'Converting'], ['enhance', 'Finishing the sound'], ['finalize', 'Saving']];
  const BUILD_STAGES = [['analyze', 'Checking your recordings'], ['extract', 'Picking your clearest phrases'],
    ['convert', 'Learning your voice'], ['finalize', 'Saving']];
  const FT_STAGES = [['analyze', 'Preparing your recordings'], ['convert', 'Training'],
    ['enhance', 'Testing the result'], ['finalize', 'Saving']];

  function plainDetail(d) {
    const m = /^Step (\d+)\/(\d+)/.exec(d || '');
    return m ? `Step ${m[1]} of ${m[2]}` : (d || '');
  }
  function renderStages(container, job, stages, doneText) {
    container.classList.remove('hidden', 'done', 'error');
    container.innerHTML = '';
    const idx = stages.findIndex(s => s[0] === job.stage);
    const finished = job.status === 'done';
    const top = el('div', 'progress-top');
    let title;
    if (finished) title = doneText;
    else if (job.status === 'error') title = 'Something went wrong';
    else if (job.status === 'cancelled') title = 'Stopped';
    else if (job.status === 'queued') title = 'Waiting for the previous job to finish';
    else title = idx >= 0 ? `${stages[idx][1]}…` : 'Starting…';
    top.append(el('span', 'progress-title', title));
    if (!finished && job.status === 'running') top.append(el('span', 'progress-pct', `${job.progress || 0}%`));
    container.append(top);
    const bar = el('div', 'bar'); const fill = el('div', 'fill'); fill.style.width = `${finished ? 100 : job.progress || 0}%`; bar.append(fill);
    container.append(bar);
    const list = el('ul', 'stage-list');
    stages.forEach(([key, label], i) => {
      const done = finished || i < idx;
      const current = !finished && key === job.stage && job.status === 'running';
      list.append(el('li', done ? 'done' : current ? 'current' : null, label));
    });
    container.append(list);
    if (job.status === 'error') { container.classList.add('error'); container.append(el('p', 'progress-detail', job.error || 'Unknown error.')); }
    else if (finished) container.classList.add('done');
    else if (job.status === 'running' && job.detail) container.append(el('p', 'progress-detail', plainDetail(job.detail)));
  }

  // ------------------------------------------------------------ waveform
  const audioCtx = () => (window._ectx = window._ectx || new (window.AudioContext || window.webkitAudioContext)());
  async function drawWave(container, blob) {
    const canvas = container.querySelector('canvas');
    try {
      const buf = await blob.arrayBuffer();
      const audio = await audioCtx().decodeAudioData(buf.slice(0));
      const data = audio.getChannelData(0);
      container.classList.remove('empty');  // show the canvas before measuring it
      const dpr = window.devicePixelRatio || 1;
      canvas.dataset.h = canvas.dataset.h || canvas.getAttribute('height');  // design height, not the scaled one
      const w = canvas.clientWidth || 600, h = parseInt(canvas.dataset.h, 10) || 80;
      canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = `${h}px`;
      const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = getComputedStyle(container).getPropertyValue('--wave').trim() || '#0e6a6e';
      const bars = Math.floor(w / 3), step = Math.max(1, Math.floor(data.length / bars));
      let peak = 1e-4;
      const levels = [];
      for (let b = 0; b < bars; b++) {
        let sum = 0; for (let j = 0; j < step; j++) { const v = data[b * step + j] || 0; sum += v * v; }
        const rms = Math.sqrt(sum / step); levels.push(rms); if (rms > peak) peak = rms;
      }
      levels.forEach((v, b) => { const bh = Math.max(2, (v / peak) * (h - 4)); ctx.fillRect(b * 3, (h - bh) / 2, 2, bh); });
    } catch (e) { /* codec not decodable by the browser: keep the player only */ }
  }
  function setPlayer(container, blob) {
    const audio = container.querySelector('audio');
    const url = URL.createObjectURL(blob); state.blobUrls.push(url);
    audio.src = url;
    drawWave(container, blob);
  }
  function clearPlayer(container) {
    container.querySelector('audio').removeAttribute('src');
    const c = container.querySelector('canvas'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
    container.classList.add('empty');
  }

  const ENGINE_TEXT = {
    seedvc_v1: ['Standard (recommended)', 'Fast and natural. Keeps the timing, pauses and emotion of the original.'],
    seedvc_v2: ['Alternative', 'A different, larger model. Slower, and pitch can\'t be changed. Try it if the standard one struggles with a recording.'],
  };

  // ------------------------------------------------------------ system
  async function loadSystem() {
    try {
      state.system = await api('/system');
      const s = state.system;
      const pill = $('sysStatus');
      pill.textContent = s.device === 'cuda' ? 'Ready, using the graphics card' : 'Ready, running on the processor (slower)';
      pill.className = `status-dot ${s.device === 'cuda' ? 'ok' : 'slow'}`;
      $('consentText').textContent = s.consent_statement;
      $('ftConsentText').textContent = s.finetune?.consent_statement || '';
      const fs = $('ftPreset'); fs.innerHTML = '';
      const names = { quick: 'Quick', standard: 'Standard', thorough: 'Thorough' };
      Object.entries(s.finetune?.presets || {}).forEach(([k, steps]) => {
        const mins = Math.round(steps * 0.47 / 60 + 1.5);
        const o = el('option', null, `${names[k] || k}, about ${mins} minutes`); o.value = k; o.selected = k === 'standard'; fs.append(o);
      });
      if (!s.separation_available) {
        document.querySelectorAll('input[name="background"]').forEach(r => { r.disabled = r.value !== 'off'; r.checked = r.value === 'off'; });
      }
      const es = $('engineSelect'); es.innerHTML = '';
      s.engines.forEach(e => {
        const o = el('option', null, `${(ENGINE_TEXT[e.id] || [e.name])[0]}${e.available ? '' : ' (not installed)'}`);
        o.value = e.id; o.disabled = !e.available; if (e.id === s.default_engine) o.selected = true; es.append(o);
      });
      updateEngineDesc(); updateBackgroundHint(); updateQualityHint();
      renderStorage(s.storage);
    } catch (e) {
      $('sysStatus').textContent = "Can't reach the app"; $('sysStatus').className = 'status-dot bad';
      toast(e.message, true);
    }
  }
  function updateEngineDesc() {
    const e = (state.system?.engines || []).find(x => x.id === $('engineSelect').value);
    $('engineDesc').textContent = e ? (ENGINE_TEXT[e.id] || [null, e.description])[1] : '';
    const caps = e ? e.capabilities : {};
    $('pitch').disabled = !caps.pitch; $('speed').disabled = !caps.speed; $('similarity').disabled = !caps.similarity;
  }
  async function renderStorage(st) {
    try {
      st = st || (await api('/system')).storage;
      $('storageInfo').textContent = `Stored in ${st.data_dir}: ${plural(st.profiles, 'voice', 'voices')}, ${plural(st.sources, 'recording', 'recordings')} to convert and ${plural(st.jobs, 'result', 'results')}, ${fmtBytes(st.bytes)} in total. The voice models take ${fmtBytes(st.models_bytes)}.`;
    } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------ your voice (profiles)
  const currentProfile = () => state.profiles.find(p => p.id === state.profileId) || null;

  async function loadProfiles(keepId) {
    state.profiles = await api('/profiles');
    if (keepId) state.profileId = keepId;
    if (!state.profiles.find(p => p.id === state.profileId)) state.profileId = state.profiles[0]?.id || null;
    renderProfileSelect();
    renderProfile();
  }
  function renderProfileSelect() {
    const sel = $('profileSelect'); sel.innerHTML = '';
    state.profiles.forEach(p => {
      const o = el('option', null, p.status === 'ready' ? p.name : `${p.name} (not learned yet)`);
      o.value = p.id; o.selected = p.id === state.profileId; sel.append(o);
    });
  }
  function showCreateForm(show) {
    $('newProfileForm').classList.toggle('hidden', !show);
    renderProfile();
    if (show) $('newProfileName').focus();
  }

  const speechOf = (p) => p.samples.reduce((a, s) => a + (s.quality?.speech_duration || 0), 0);
  function voiceStateText(p) {
    const speech = speechOf(p);
    const n = p.samples.length;
    const building = busy(state.buildJob) && state.buildJob.profile_id === p.id;
    if (building || p.status === 'building') return ['busy', 'Learning your voice…', 'This takes about a minute. You can add your friend\'s recording meanwhile.'];
    if (p.status === 'error') return ['error', "Couldn't learn your voice", `${p.error || 'Unknown error.'} Try again, or replace the recording that has problems.`];
    if (!n) return ['todo', 'Add recordings of you', 'Upload or record 1–5 minutes of you talking. Several short recordings are fine.'];
    if (p.status === 'needs_build') {
      const more = speech < 60 ? ` A little more helps: 1–5 minutes works best.` : '';
      return ['todo', 'Ready to learn your voice', `You have ${fmtTime(speech)} of speech from ${plural(n, 'recording', 'recordings')}.${more} Then click Learn my voice.`];
    }
    const when = p.built_at ? new Date(p.built_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '';
    return ['ready', 'Your voice is ready', `Learned from ${fmtTime(speech)} of speech in ${plural(n, 'recording', 'recordings')}${when ? `, on ${when}` : ''}.`];
  }

  function renderProfile() {
    const p = currentProfile();
    const creating = !$('newProfileForm').classList.contains('hidden');
    $('voiceEmpty').classList.toggle('hidden', !!p || creating);
    $('profilePicker').classList.toggle('hidden', !state.profiles.length);
    $('profileArea').classList.toggle('hidden', !p || creating);
    $('delProfileBtn').disabled = !p;
    updateFlow();
    if (!p) return;

    const [cls, title, text] = voiceStateText(p);
    $('voiceState').className = `voice-state ${cls}`;
    $('voiceStateTitle').textContent = title;
    $('profileStatus').textContent = text;
    const speech = speechOf(p);
    $('speechProgress').classList.toggle('hidden', p.status === 'ready');
    $('speechNow').textContent = `${fmtTime(speech)} of speech`;
    $('speechMeter').classList.toggle('good', speech >= 60);
    $('speechMeter').firstElementChild.style.width = `${Math.min(100, speech / 180 * 100)}%`;

    // recordings of you
    const ul = $('sampleList'); ul.innerHTML = '';
    p.samples.forEach(s => {
      const li = el('li', 'rec-item');
      const main = el('div', 'rec-main');
      main.append(el('span', 'rec-name', s.original_name), el('span', 'rec-len', fmtTime(s.duration)));
      const actions = el('div', 'rec-actions');
      const play = el('button', 'icon-btn', 'Play'); play.type = 'button';
      play.setAttribute('aria-label', `Play ${s.original_name}`);
      play.onclick = async () => { try { playInline(li, await api(`/profiles/${p.id}/samples/${s.id}/audio`, { blob: true })); } catch (e) { toast(e.message, true); } };
      const del = el('button', 'icon-btn', 'Delete'); del.type = 'button';
      del.setAttribute('aria-label', `Delete ${s.original_name}`);
      del.onclick = async () => {
        if (!confirm(`Delete "${s.original_name}"? You will need to learn your voice again afterwards.`)) return;
        try { await api(`/profiles/${p.id}/samples/${s.id}`, { method: 'DELETE' }); await loadProfiles(p.id); renderStorage(); toast('Recording deleted.'); } catch (e) { toast(e.message, true); }
      };
      actions.append(play, del); main.append(actions); li.append(main);

      const tags = el('div', 'rec-tags');
      const prep = p.status === 'ready' ? s.prepared : null;
      if (prep && !prep.used_for_reference && !prep.used_for_identity) {
        tags.append(el('span', 'tag poor', 'Not used'));
      } else {
        tags.append(qualityTag(s.quality, true));
      }
      if (prep && prep.separated) tags.append(el('span', 'tag info', 'Music removed'));
      li.append(tags);
      if (prep && !prep.used_for_reference && !prep.used_for_identity) {
        li.append(el('p', 'hint', 'Left out because it is much noisier than your other recordings. It is used again if you delete the others.'));
      } else if (issuesOf(s.quality, true).length) {
        li.append(issueList(s.quality, true));
      }
      ul.append(li);
    });
    $('recTitle').textContent = p.samples.length ? `Recordings of you (${p.samples.length})` : 'Recordings of you';
    // once the voice is learned the list gets out of the way; reopened when something needs doing
    const stateKey = `${p.id}:${p.status}`;
    if (state.recBlockKey !== stateKey) { $('recBlock').open = p.status !== 'ready'; state.recBlockKey = stateKey; }

    // learn button
    const building = busy(state.buildJob) && state.buildJob.profile_id === p.id;
    const btn = $('buildBtn');
    btn.classList.toggle('hidden', !p.samples.length);
    btn.disabled = building || p.status === 'building';
    btn.textContent = p.status === 'ready' ? 'Learn my voice again' : 'Learn my voice';
    btn.classList.toggle('primary', p.status !== 'ready');
    btn.classList.toggle('quiet', p.status === 'ready');
    if (!building && p.status !== 'building' && state.buildJob?.status === 'done') $('buildStatus').classList.add('hidden');

    const notes = $('buildNotes'); notes.innerHTML = '';
    (p.status === 'ready' ? (p.build_notes || []) : []).forEach(n => notes.append(el('li', null, n)));
    notes.classList.toggle('hidden', !notes.children.length);
    renderAnalysis(p);
    renderFinetune(p);
  }
  function playInline(li, blob) {
    let a = li.querySelector('audio');
    if (!a) { a = document.createElement('audio'); a.controls = true; li.append(a); }
    const url = URL.createObjectURL(blob); state.blobUrls.push(url); a.src = url; a.play().catch(() => {});
  }

  function renderAnalysis(p) {
    const box = $('profileAnalysis');
    const a = p.analysis;
    const show = !!(a && p.status === 'ready');
    $('voiceDetails').classList.toggle('hidden', !show);
    box.innerHTML = '';
    if (!show) return;
    const dl = el('dl', 'traits');
    const trait = (k, v) => { if (v) dl.append(el('dt', null, k), el('dd', null, v)); };
    if (a.pitch?.available) trait('Pitch', `${a.pitch.level}, ${a.pitch.variability}`);
    if (a.timbre?.available) trait('Tone', a.timbre.brightness);
    if (a.texture?.available) trait('Texture', a.texture.quality);
    if (a.rhythm?.available) trait('Pace', `${a.rhythm.pace}, ${a.rhythm.regularity} rhythm`);
    if (a.pauses?.available) trait('Pauses', a.pauses.style);
    if (a.timbre?.available) trait('Pronunciation', a.timbre.articulation);
    if (a.loudness?.available) trait('Loudness', a.loudness.dynamics);
    box.append(dl);
    const eng = p.engines?.[state.system?.default_engine] || Object.values(p.engines || {})[0];
    const low = (eng?.meta?.sample_consistency || []).filter(c => c < 0.6).length;
    if (low) box.append(el('p', 'hint', `${plural(low, 'recording sounds', 'recordings sound')} quite different from the rest (another person, or a lot of noise?). Removing it can make the result sound more like you.`));
    const refBtn = el('button', 'btn quiet small', 'Play the clip the app learned from'); refBtn.type = 'button';
    refBtn.onclick = async () => { try { playInline(box, await api(`/profiles/${p.id}/reference`, { blob: true })); } catch (e) { toast(e.message, true); } };
    box.append(refBtn);
  }

  $('startVoiceBtn').onclick = () => showCreateForm(true);
  $('newProfileBtn').onclick = () => showCreateForm(true);
  $('cancelProfileBtn').onclick = () => showCreateForm(false);
  $('consentBox').onchange = () => { $('createProfileBtn').disabled = !$('consentBox').checked; };
  $('newProfileForm').onsubmit = async (e) => {
    e.preventDefault();
    try {
      const p = await api('/profiles', { method: 'POST', json: { name: $('newProfileName').value, consent: $('consentBox').checked } });
      $('consentBox').checked = false; $('createProfileBtn').disabled = true;
      $('newProfileForm').classList.add('hidden');
      await loadProfiles(p.id);
      toast('Voice created. Now add recordings of you.');
    } catch (err) { toast(err.message, true); }
  };
  $('profileSelect').onchange = () => { state.profileId = $('profileSelect').value; renderProfile(); };

  // tabs (upload / record)
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => { x.classList.toggle('active', x === t); x.setAttribute('aria-selected', String(x === t)); });
    $('tab-upload').classList.toggle('hidden', t.dataset.tab !== 'upload');
    $('tab-record').classList.toggle('hidden', t.dataset.tab !== 'record');
  });

  // uploads
  function wireDrop(zone, input, onFiles) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); });
    input.addEventListener('change', () => { if (input.files.length) onFiles(input.files); input.value = ''; });
  }
  function showUpload(id, frac) {
    const box = $(id); box.classList.toggle('hidden', frac == null);
    if (frac != null) { box.querySelector('.fill').style.width = `${Math.round(frac * 100)}%`; box.querySelector('.label-text').textContent = frac >= 1 ? 'Checking the sound quality…' : 'Uploading…'; }
  }
  async function addSamples(files, kind) {
    const p = currentProfile(); if (!p) return;
    const fd = new FormData();
    Array.from(files).forEach(f => fd.append('files', f, f.name));
    fd.append('kind', kind || 'upload');
    try {
      showUpload('sampleUpload', 0);
      const r = await upload(`/profiles/${p.id}/samples`, fd, (f) => showUpload('sampleUpload', f));
      showUpload('sampleUpload', null);
      r.errors.forEach(e => toast(`${e.file}: ${e.error}`, true));
      const poor = r.added.filter(s => s.quality && s.quality.issues.some(i => i.severity === 'poor'));
      if (poor.length) toast('Added, but some recordings have sound problems. See the notes under each one.', true);
      else if (r.added.length) toast(`Added ${plural(r.added.length, 'recording', 'recordings')}.`);
      await loadProfiles(p.id); renderStorage();
    } catch (e) { showUpload('sampleUpload', null); toast(e.message, true); }
  }
  wireDrop($('sampleDrop'), $('sampleInput'), (files) => addSamples(files, 'upload'));
  $('samplePick').onclick = () => $('sampleInput').click();

  // ------------------------------------------------------------ recorder
  const rec = { stream: null, mr: null, chunks: [], start: 0, timer: null, ctx: null, analyser: null, raf: null };
  function drawMeter(level) {
    const c = $('meter'), ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    const cs = getComputedStyle(document.documentElement);
    ctx.fillStyle = level > 0.9 ? cs.getPropertyValue('--bad') : level > 0.6 ? cs.getPropertyValue('--warn') : cs.getPropertyValue('--teal');
    ctx.fillRect(0, 0, Math.min(c.width, Math.round(c.width * level)), c.height);
  }
  async function startRecording() {
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) { toast('The browser could not use your microphone. Allow microphone access for this page and try again.', true); return; }
    const mime = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'].find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
    rec.mr = new MediaRecorder(rec.stream, mime ? { mimeType: mime, audioBitsPerSecond: 192000 } : undefined);
    rec.chunks = [];
    rec.mr.ondataavailable = (e) => { if (e.data.size) rec.chunks.push(e.data); };
    rec.mr.onstop = () => {
      state.recBlob = new Blob(rec.chunks, { type: rec.mr.mimeType || 'audio/webm' });
      $('recAudio').src = URL.createObjectURL(state.recBlob);
      $('recPreview').classList.remove('hidden');
      const secs = (Date.now() - rec.start) / 1000;
      $('recHint').textContent = secs < 60 ? `Recorded ${fmtTime(secs)}. That works, but 1–5 minutes in total gives a much better result.` : `Recorded ${fmtTime(secs)}.`;
      rec.stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(rec.raf); drawMeter(0);
      if (rec.ctx) { rec.ctx.close().catch(() => {}); rec.ctx = null; }
    };
    rec.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = rec.ctx.createMediaStreamSource(rec.stream);
    rec.analyser = rec.ctx.createAnalyser(); rec.analyser.fftSize = 1024; src.connect(rec.analyser);
    const buf = new Float32Array(rec.analyser.fftSize);
    const tick = () => {
      rec.analyser.getFloatTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      drawMeter(Math.min(1, Math.sqrt(sum / buf.length) * 6));
      rec.raf = requestAnimationFrame(tick);
    };
    tick();
    rec.start = Date.now();
    rec.timer = setInterval(() => { $('recTime').textContent = fmtTime((Date.now() - rec.start) / 1000); }, 250);
    rec.mr.start(250);
    $('recBtn').textContent = 'Stop recording'; $('recBtn').classList.add('recording');
    $('recPreview').classList.add('hidden'); $('recHint').textContent = 'Recording. Talk naturally, then click Stop recording.';
  }
  function stopRecording() {
    clearInterval(rec.timer);
    if (rec.mr && rec.mr.state !== 'inactive') rec.mr.stop();
    $('recBtn').textContent = 'Start recording'; $('recBtn').classList.remove('recording');
  }
  $('recBtn').onclick = () => { if (rec.mr && rec.mr.state === 'recording') stopRecording(); else startRecording(); };
  $('recDiscardBtn').onclick = () => { state.recBlob = null; $('recPreview').classList.add('hidden'); $('recTime').textContent = '0:00'; $('recHint').textContent = ''; };
  $('recAddBtn').onclick = async () => {
    if (!state.recBlob) return;
    const ext = state.recBlob.type.includes('ogg') ? 'ogg' : state.recBlob.type.includes('mp4') ? 'm4a' : 'webm';
    const name = `recording-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.${ext}`;
    await addSamples([new File([state.recBlob], name, { type: state.recBlob.type })], 'recording');
    $('recDiscardBtn').click();
  };

  // learn my voice
  $('buildBtn').onclick = async () => {
    const p = currentProfile(); if (!p) return;
    try {
      const job = await api(`/profiles/${p.id}/build`, { method: 'POST', json: { engine: $('engineSelect').value } });
      state.buildJob = job; renderStages($('buildStatus'), job, BUILD_STAGES, 'Your voice is ready.');
      renderProfile();
      clearInterval(state.buildPoll);
      state.buildPoll = setInterval(async () => {
        try {
          const j = await api(`/jobs/${job.id}`);
          state.buildJob = j; renderStages($('buildStatus'), j, BUILD_STAGES, 'Your voice is ready.');
          if (!busy(j)) {
            clearInterval(state.buildPoll); await loadProfiles(p.id); renderStorage();
            if (j.status === 'done') toast('Your voice is ready.'); else if (j.status === 'error') toast(j.error, true);
          }
        } catch (e) { clearInterval(state.buildPoll); toast(e.message, true); }
      }, 800);
    } catch (e) { toast(e.message, true); }
  };

  // ------------------------------------------------------------ training (opt-in)
  const ftRunning = (p) => !!(p && state.ftJob && state.ftJob.profile_id === p.id && busy(state.ftJob));
  function renderFinetune(p) {
    const ft = state.system?.finetune;
    const show = !!(p && p.status === 'ready' && ft);
    $('finetunePanel').classList.toggle('hidden', !show);
    const hasModel = !!(p && p.finetune && p.finetune.status === 'ready');
    $('useFinetunedRow').classList.toggle('hidden', !hasModel);
    if (!show) return;
    const allowed = !!(p.allow_training && p.training_consent && p.training_consent.confirmed);
    $('ftConsentBox').classList.toggle('hidden', allowed);
    $('ftControls').classList.toggle('hidden', !allowed);
    const running = ftRunning(p);
    $('ftStartBtn').disabled = running || !ft.gpu;
    $('ftStartBtn').title = ft.gpu ? '' : 'Training needs an NVIDIA graphics card.';
    $('ftStartBtn').textContent = hasModel ? 'Train again' : 'Start training';
    $('ftCancelBtn').classList.toggle('hidden', !running);
    $('ftDeleteBtn').classList.toggle('hidden', !hasModel);
    $('ftDeleteBtn').disabled = running;
    $('ftWithdrawBtn').disabled = running;
    if (running) $('finetunePanel').open = true;
    renderFinetuneResult(p.finetune);
  }
  function renderFinetuneResult(ft) {
    const box = $('ftResult'); box.innerHTML = '';
    if (!ft) return;
    const when = (t) => (t ? new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '');
    if (ft.status === 'ready') {
      box.append(el('p', 'hint', `Your trained model is in use (trained on ${when(ft.trained_at)} from ${Math.round(ft.train_seconds)} seconds of your speech).` +
        (ft.evaluated ? '' : ' It has not been tested yet, because there was no recording of another person to test with. Add a recording in step 1 and train again to check it.')));
    } else {
      box.append(el('p', 'hint', `The training on ${when(ft.trained_at)} did not beat the standard model, so the standard model stays in use. More recordings, or cleaner ones, usually help.`));
    }
    if (ft.last_run) box.append(el('p', 'hint', `A newer training run (${when(ft.last_run.trained_at)}) was not better, so it was discarded.`));
    const shown = ft.last_run || ft;
    const cands = shown.candidates || [];
    if (!cands.length || cands[0].speaker_sim == null) return;
    const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
    const wrap = el('div', 'table-wrap'); const t = el('table', 'ft-table');
    const hr = el('tr'); ['Model', 'Sounds like you', 'Words lost', 'Natural (of 5)'].forEach(h => hr.append(el('th', null, h))); t.append(hr);
    cands.forEach(c => {
      const inUse = (shown.status === 'ready' && c.step === shown.chosen_step) || (shown.status !== 'ready' && c.step === 0);
      const tr = el('tr', inUse ? 'chosen' : null);
      [`${c.step === 0 ? 'Standard' : `Trained, step ${c.step}`}${inUse ? ' (in use)' : ''}`, pct(c.speaker_sim), pct(c.cer), c.utmos == null ? '–' : c.utmos.toFixed(1)].forEach(v => tr.append(el('td', null, v)));
      t.append(tr);
    });
    wrap.append(t); box.append(wrap);
    box.append(el('p', 'hint', `Tested by converting ${plural((shown.probes || []).length, 'recording', 'recordings')} of another person and comparing with ${plural(shown.heldout_clips, 'clip', 'clips')} of your voice that were kept out of training.`));
  }
  $('ftConsent').onchange = () => { $('ftAllowBtn').disabled = !$('ftConsent').checked; };
  $('ftAllowBtn').onclick = async () => {
    const p = currentProfile(); if (!p || !$('ftConsent').checked) return;
    try { await api(`/profiles/${p.id}/training-consent`, { method: 'POST', json: { allow: true } }); $('ftConsent').checked = false; $('ftAllowBtn').disabled = true; await loadProfiles(p.id); toast('Training allowed for this voice.'); } catch (e) { toast(e.message, true); }
  };
  $('ftWithdrawBtn').onclick = async () => {
    const p = currentProfile(); if (!p || !confirm('Stop allowing training for this voice? This also deletes its trained model.')) return;
    try { await api(`/profiles/${p.id}/training-consent`, { method: 'POST', json: { allow: false } }); await loadProfiles(p.id); renderStorage(); toast('Training is no longer allowed. The trained model was deleted.'); } catch (e) { toast(e.message, true); }
  };
  $('ftDeleteBtn').onclick = async () => {
    const p = currentProfile(); if (!p || !confirm('Delete the trained model? Conversions go back to the standard model.')) return;
    try { await api(`/profiles/${p.id}/finetune`, { method: 'DELETE' }); await loadProfiles(p.id); renderStorage(); toast('Trained model deleted.'); } catch (e) { toast(e.message, true); }
  };
  $('ftStartBtn').onclick = async () => {
    const p = currentProfile(); if (!p) return;
    try { state.ftJob = await api(`/profiles/${p.id}/finetune`, { method: 'POST', json: { preset: $('ftPreset').value } }); pollFinetune(); } catch (e) { toast(e.message, true); }
  };
  $('ftCancelBtn').onclick = async () => { if (state.ftJob) { try { await api(`/jobs/${state.ftJob.id}/cancel`, { method: 'POST' }); } catch (e) { toast(e.message, true); } } };
  function pollFinetune() {
    renderStages($('ftStatus'), state.ftJob, FT_STAGES, 'Training finished.');
    renderFinetune(currentProfile());
    clearInterval(state.ftPoll);
    state.ftPoll = setInterval(async () => {
      try {
        const j = await api(`/jobs/${state.ftJob.id}`);
        state.ftJob = j; renderStages($('ftStatus'), j, FT_STAGES, j.done_label || 'Training finished.');
        if (!busy(j)) {
          clearInterval(state.ftPoll); await loadProfiles(j.profile_id); renderStorage();
          if (j.status === 'done') toast(j.done_label || 'Training finished.'); else if (j.status === 'error') toast(j.error, true);
        }
      } catch (e) { clearInterval(state.ftPoll); toast(e.message, true); }
    }, 1500);
  }

  // ------------------------------------------------------------ step 1: the recording to convert
  async function loadSources(keepId) {
    state.sources = await api('/sources');
    if (keepId) state.sourceId = keepId;
    if (!state.sources.find(s => s.id === state.sourceId)) state.sourceId = state.sources[0]?.id || null;
    renderSources();
  }
  function selectSource(id) {
    if (state.sourceId === id) return;
    state.sourceId = id; renderSources(); loadSourcePreview();
  }
  function renderSources() {
    const ul = $('sourceList'); ul.innerHTML = '';
    state.sources.forEach(s => {
      const li = el('li', `rec-item ${s.id === state.sourceId ? 'selected' : ''}`);
      const main = el('div', 'rec-main');
      const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'source'; radio.checked = s.id === state.sourceId;
      radio.setAttribute('aria-label', `Convert ${s.original_name}`);
      radio.onchange = () => selectSource(s.id);
      main.append(radio, el('span', 'rec-name', s.original_name), el('span', 'rec-len', fmtTime(s.duration)));
      const actions = el('div', 'rec-actions');
      const play = el('button', 'icon-btn', 'Play'); play.type = 'button';
      play.onclick = async () => { try { playInline(li, await api(`/sources/${s.id}/audio`, { blob: true })); } catch (e) { toast(e.message, true); } };
      const del = el('button', 'icon-btn', 'Delete'); del.type = 'button';
      del.onclick = async () => { if (!confirm(`Delete "${s.original_name}"?`)) return; try { await api(`/sources/${s.id}`, { method: 'DELETE' }); await loadSources(); renderStorage(); toast('Recording deleted.'); } catch (e) { toast(e.message, true); } };
      actions.append(play, del); main.append(actions); li.append(main);
      const tags = el('div', 'rec-tags'); tags.append(qualityTag(s.quality, false)); li.append(tags);
      if (s.quality && s.quality.issues.length) li.append(issueList(s.quality));
      li.onclick = (e) => { if (['BUTTON', 'AUDIO', 'INPUT'].includes(e.target.tagName)) return; selectSource(s.id); };
      ul.append(li);
    });
    $('delSourceBtn').disabled = !state.sourceId;
    $('sourceDrop').classList.toggle('compact', state.sources.length > 0);
    updateFlow(); updateQualityHint();
  }
  async function addSource(files) {
    const f = files[0]; if (!f) return;
    const fd = new FormData(); fd.append('file', f, f.name);
    try {
      showUpload('sourceUpload', 0);
      const s = await upload('/sources', fd, (fr) => showUpload('sourceUpload', fr));
      showUpload('sourceUpload', null);
      await loadSources(s.id); renderStorage();
      if (s.quality.issues.some(i => i.severity === 'poor')) toast('Added, but this recording has sound problems, so the result may suffer. See the notes under it.', true);
      else toast('Recording added.');
      loadSourcePreview();
    } catch (e) { showUpload('sourceUpload', null); toast(e.message, true); }
  }
  wireDrop($('sourceDrop'), $('sourceInput'), addSource);
  $('sourcePick').onclick = () => $('sourceInput').click();
  async function loadSourcePreview() {
    if (!state.sourceId) { clearPlayer($('playerSource')); return; }
    try { setPlayer($('playerSource'), await api(`/sources/${state.sourceId}/audio`, { blob: true })); } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------ step 2: how it sounds
  const ctl = { similarity: $('similarity'), pitch: $('pitch'), speed: $('speed'), energy: $('energy') };
  const radioValue = (name) => (document.querySelector(`input[name="${name}"]:checked`) || {}).value;
  function syncOutputs() {
    $('similarityOut').textContent = ctl.similarity.value;
    const pv = +ctl.pitch.value;
    $('pitchOut').textContent = pv === 0 ? 'Natural' : `${pv > 0 ? '+' : '−'}${Math.abs(pv)} semitone${Math.abs(pv) === 1 ? '' : 's'}`;
    $('speedOut').textContent = `${(+ctl.speed.value).toFixed(2)}×`;
    $('energyOut').textContent = `${(+ctl.energy.value).toFixed(2)}×`;
  }
  Object.values(ctl).forEach(i => i.addEventListener('input', () => { syncOutputs(); updateQualityHint(); }));

  const RTF = { draft: 0.12, standard: 0.16, high: 0.23 };  // measured on an RTX 4050 laptop (README)
  function updateQualityHint() {
    const q = radioValue('quality') || 'standard';
    const what = { draft: 'Fastest. Good for checking how it will sound.', standard: 'Natural sound. Best for most videos.', high: 'Slightly more detail. Takes about twice as long.' }[q];
    const src = state.sources.find(s => s.id === state.sourceId);
    let time = '';
    if (src && state.system?.device === 'cuda') {
      const secs = Math.max(2, Math.round(src.duration * RTF[q] / (+ctl.speed.value || 1)));
      time = ` About ${secs} seconds for this ${fmtDur(src.duration)} recording.`;
    } else if (src && state.system && state.system.device !== 'cuda') {
      time = ' Without a graphics card this takes a few minutes per minute of audio.';
    }
    $('qualityHint').textContent = what + time;
  }
  function updateBackgroundHint() {
    const v = radioValue('background');
    $('backgroundHint').textContent = state.system && !state.system.separation_available
      ? 'Music separation is turned off on this computer, so the recording is converted as it is.'
      : { keep: 'If the app hears music or loud background sound, it lifts the voice out, converts it, and puts the music back underneath.',
          remove: 'You get only your converted voice, with the music and background taken out.',
          off: 'The recording is converted as it is. Choose this for clean recordings without music.' }[v];
  }
  document.querySelectorAll('input[name="quality"]').forEach(r => r.addEventListener('change', updateQualityHint));
  document.querySelectorAll('input[name="background"]').forEach(r => r.addEventListener('change', updateBackgroundHint));
  $('resetBtn').onclick = () => {
    const d = state.system?.defaults || { similarity: 65, pitch_semitones: 0, speed: 1, energy: 1, quality: 'standard', denoise: 'auto' };
    ctl.similarity.value = d.similarity; ctl.pitch.value = d.pitch_semitones; ctl.speed.value = d.speed; ctl.energy.value = d.energy;
    document.querySelector(`input[name="quality"][value="${d.quality}"]`).checked = true;
    const bg = document.querySelector(`input[name="background"][value="${state.system?.separation_available === false ? 'off' : 'keep'}"]`);
    if (bg) bg.checked = true;
    $('denoise').value = d.denoise; $('preserveDynamics').checked = true; $('useFinetuned').checked = d.use_finetuned !== false;
    syncOutputs(); updateQualityHint(); updateBackgroundHint();
    toast('Back to the defaults.');
  };
  $('engineSelect').onchange = updateEngineDesc;
  syncOutputs();

  // which steps are usable right now
  function updateFlow() {
    const p = currentProfile();
    const voiceReady = !!(p && p.status === 'ready');
    const converting = busy(state.job);
    const ready = voiceReady && !!state.sourceId && !converting;
    $('convertBtn').disabled = !ready;
    $('convertBtn').textContent = converting ? 'Converting…' : 'Convert to my voice';
    $('convertHint').textContent = converting ? ''
      : !p ? 'Create your voice first, in the panel on the left.'
      : !voiceReady ? 'Learn your voice first, in the panel on the left.'
      : !state.sourceId ? "Add your friend's recording in step 1." : '';
    $('stepSource').classList.toggle('done', !!state.sourceId);
    const optionsLocked = !voiceReady || !state.sourceId;
    $('stepOptions').classList.toggle('locked', optionsLocked);
    $('stepOptions').classList.toggle('done', !!state.result && !optionsLocked);
    $('sec-preview').classList.toggle('locked', !state.result);
  }

  // ------------------------------------------------------------ convert
  $('convertBtn').onclick = async () => {
    const p = currentProfile(); if (!p || !state.sourceId) return;
    const bg = radioValue('background') || 'keep';
    const settings = {
      similarity: +ctl.similarity.value, pitch_semitones: +ctl.pitch.value, speed: +ctl.speed.value, energy: +ctl.energy.value,
      quality: radioValue('quality') || 'standard', denoise: $('denoise').value, preserve_dynamics: $('preserveDynamics').checked,
      separate: { keep: 'auto', remove: 'on', off: 'off' }[bg], keep_background: bg !== 'remove',
      use_finetuned: $('useFinetuned').checked,
    };
    try {
      const job = await api('/convert', { method: 'POST', json: { profile_id: p.id, source_id: state.sourceId, engine: $('engineSelect').value, settings } });
      state.job = job; state.result = null;
      renderStages($('convertStatus'), job, STAGES, 'Converted.');
      setExportEnabled(false); clearPlayer($('playerOut')); $('resultMeta').innerHTML = '';
      updateFlow(); loadSourcePreview();
      clearInterval(state.poll);
      state.poll = setInterval(pollJob, 800);
    } catch (e) { toast(e.message, true); }
  };
  async function pollJob() {
    if (!state.job) return;
    try {
      const j = await api(`/jobs/${state.job.id}`);
      state.job = j; renderStages($('convertStatus'), j, STAGES, 'Converted.');
      if (!busy(j)) {
        clearInterval(state.poll); updateFlow();
        if (j.status === 'done') { await showResult(j, true); toast('Converted. Listen to it in step 3.'); }
        else if (j.status === 'error') toast(j.error, true);
      }
    } catch (e) { clearInterval(state.poll); toast(e.message, true); updateFlow(); }
  }

  // ------------------------------------------------------------ step 3: listen and download
  async function showResult(j, scroll) {
    state.result = j;
    try { setPlayer($('playerOut'), await api(`/jobs/${j.id}/audio`, { blob: true })); } catch (e) { toast(e.message, true); }
    const o = j.output || {};
    // with several voices, say which one this result was made with
    const madeWith = state.profiles.find(x => x.id === j.profile_id);
    $('playerOut').querySelector('.track-name').textContent =
      state.profiles.length > 1 && madeWith ? `In ${madeWith.name}'s voice` : 'In your voice';
    const facts = $('resultMeta'); facts.innerHTML = '';
    const fact = (strong, rest) => { const li = el('li'); if (strong) li.append(el('strong', null, strong)); if (rest) li.append(document.createTextNode(rest)); facts.append(li); };
    const vm = o.voice_match;
    if (vm && vm.converted_vs_target != null) {
      fact(`${Math.round(vm.converted_vs_target * 100)}% voice match`, ` with your voice (your friend's original scored ${Math.round(vm.source_vs_target * 100)}%).`);
    }
    const sep = o.separation;
    if (sep && sep.applied) fact(null, sep.background_remixed ? 'The music was lifted out, and put back under your voice.' : 'The music and background were removed.');
    if (o.denoised) fact(null, 'Background noise was cleaned up first.');
    if (o.finetuned) fact(null, 'Made with your trained voice model.');
    if (o.timings && o.timings.total_s) fact(null, `Took ${o.timings.total_s.toFixed(1)} seconds for ${fmtTime(o.duration)} of audio.`);
    $('resultLede').textContent = "Play both to compare. If it doesn't sound enough like you, move \"How much like you\" to the right, or add recordings of you made in a quieter room.";
    setExportEnabled(true); updateFlow(); renderStorage();
    if (scroll) $('sec-preview').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  }
  function setExportEnabled(on) {
    $('exportWav').disabled = !on; $('exportMp3').disabled = !on; $('delResultBtn').disabled = !on;
    $('downloadArea').classList.toggle('hidden', !on);
  }

  async function download(fmt) {
    if (!state.result) return;
    const name = ($('exportName').value || 'my-voice').trim();
    const btn = fmt === 'wav' ? $('exportWav') : $('exportMp3');
    const old = btn.textContent; btn.textContent = 'Preparing…'; btn.disabled = true;
    try {
      const b = await api(`/jobs/${state.result.id}/export?format=${fmt}&name=${encodeURIComponent(name)}`, { blob: true });
      const url = URL.createObjectURL(b); const a = document.createElement('a'); a.href = url; a.download = `${name}.${fmt}`; document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      toast(`Downloaded ${name}.${fmt}.`);
    } catch (e) { toast(e.message, true); }
    btn.textContent = old; btn.disabled = false;
  }
  $('exportWav').onclick = () => download('wav');
  $('exportMp3').onclick = () => download('mp3');

  // ------------------------------------------------------------ your data
  $('delResultBtn').onclick = async () => {
    if (!state.result || !confirm('Delete this converted recording from the computer?')) return;
    try {
      await api(`/jobs/${state.result.id}`, { method: 'DELETE' });
      state.result = null; setExportEnabled(false); clearPlayer($('playerOut')); $('resultMeta').innerHTML = '';
      $('convertStatus').classList.add('hidden'); updateFlow(); toast('Result deleted.'); renderStorage();
    } catch (e) { toast(e.message, true); }
  };
  $('delSourceBtn').onclick = async () => {
    const s = state.sources.find(x => x.id === state.sourceId);
    if (!s || !confirm(`Delete "${s.original_name}"?`)) return;
    try { await api(`/sources/${s.id}`, { method: 'DELETE' }); await loadSources(); loadSourcePreview(); renderStorage(); toast('Recording deleted.'); } catch (e) { toast(e.message, true); }
  };
  $('delProfileBtn').onclick = async () => {
    const p = currentProfile(); if (!p || !confirm(`Delete the voice "${p.name}", its recordings and any trained model?`)) return;
    try { await api(`/profiles/${p.id}`, { method: 'DELETE' }); await loadProfiles(); renderStorage(); toast('Voice deleted.'); } catch (e) { toast(e.message, true); }
  };
  $('delAllBtn').onclick = async () => {
    if (!confirm('Delete everything: all voices, recordings and results? This cannot be undone.')) return;
    try {
      await api('/data', { method: 'DELETE' });
      state.result = null; setExportEnabled(false); clearPlayer($('playerOut')); clearPlayer($('playerSource')); $('resultMeta').innerHTML = '';
      await loadProfiles(); await loadSources(); renderStorage();
      $('convertStatus').classList.add('hidden'); $('buildStatus').classList.add('hidden');
      toast('Everything was deleted.');
    } catch (e) { toast(e.message, true); }
  };

  // ------------------------------------------------------------ start
  (async () => {
    ['playerSource', 'playerOut'].forEach(id => $(id).classList.add('empty'));
    await loadSystem();
    try { await loadProfiles(); await loadSources(); loadSourcePreview(); } catch (e) { toast(e.message, true); }
    try {  // pick up work that was running before a page reload
      const jobs = await api('/jobs');
      const ftj = jobs.find(j => j.kind === 'finetune' && busy(j));
      if (ftj) { state.ftJob = ftj; pollFinetune(); }
      const building = jobs.find(j => j.kind === 'profile_build' && busy(j));
      if (building) { state.buildJob = building; renderStages($('buildStatus'), building, BUILD_STAGES, 'Your voice is ready.'); renderProfile(); }
      const running = jobs.find(j => j.kind === 'convert' && busy(j));
      const last = jobs.find(j => j.kind === 'convert' && j.status === 'done');
      if (running) { state.job = running; renderStages($('convertStatus'), running, STAGES, 'Converted.'); state.poll = setInterval(pollJob, 800); }
      else if (last) await showResult(last, false);
    } catch (e) { /* ignore */ }
    updateFlow();
  })();
})();
