---
layout: project_page
permalink: /

title: "[MICCAI 2025] SurgX: Neuron-Concept Association for Explainable Surgical Phase Recognition"
authors:
  - Ka Young Kim, Hyeon Bae Kim, Seong Tae Kim†
affiliations:
  - Kyung Hee University, Yongin, Republic of Korea
paper_url: https://arxiv.org/pdf/2507.15418v1
code_url: https://github.com/ailab-kyunghee/SurgX
---

<style>
  /* ===== Tighten only title↔hero and hero↔abstract ===== */

/* 레이아웃이 상단에 페이지 제목을 렌더링할 때 대비 */
.page-title,
.post-title,
h1.page-title,
h1.post-title {
  margin-bottom: 0.6rem; /* 제목과 히어로를 더 가깝게 */
}

/* 히어로 섹션 자체 간격 축소 */
.hero-section {
  padding-top: 0.6rem !important;
  padding-bottom: 0.6rem !important;
}

/* 히어로 내부 이미지와 버튼 간격도 촘촘하게 */
.hero-section .figure-hero {
  margin-bottom: 0.6rem !important;
}
.hero-section .link-blocks {
  margin-top: 0.4rem !important;
  margin-bottom: 0.6rem !important;
}

/* Abstract의 상단 여백만 줄여서 히어로와 바짝 */
.abstract-section {
  padding-top: 0.8rem !important;
}

/* 히어로와 Abstract 사이에 구분선이 있으면 여백 최소화 */
hr.section-divider {
  margin: 0.8rem auto !important;
}

/* 모바일에서는 너무 붙지 않게 살짝만 완화 */
@media (max-width: 768px) {
  .hero-section {
    padding-top: 0.8rem !important;
    padding-bottom: 0.8rem !important;
  }
  .abstract-section {
    padding-top: 1rem !important;
  }
}

/* Pretendard Font 설정 */
@font-face {
  font-family: 'Pretendard';
  src: url('./static/font/Pretendard-Regular.otf') format('opentype');
  font-weight: 400;
  font-style: normal;
}
@font-face {
  font-family: 'Pretendard';
  src: url('./static/font/Pretendard-Medium.otf') format('opentype');
  font-weight: 500;
  font-style: normal;
}
@font-face {
  font-family: 'Pretendard';
  src: url('./static/font/Pretendard-Black.otf') format('opentype');
  font-weight: 900;
  font-style: normal;
}

/* 전역 기본 폰트 적용 */
body {
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* --- Desktop 확대/모바일 풀폭 설정 --- */
@media screen and (min-width: 1216px) {
  .narrow-container {
    max-width: 1200px;
    margin: 0 auto;
  }
}
@media screen and (min-width: 1408px) {
  .narrow-container {
    max-width: 1280px;
    margin: 0 auto;
  }
}

/* 기본: 좌측 정렬 */
body,
.narrow-container,
.narrow-container .content,
ul, ol, li, p {
  text-align: left;
}

/* 제목도 좌측 */
h1, h2, h3, h4, h5, h6 {
  text-align: left;
}

/* Hero, Abstract 섹션은 중앙 정렬 강제 */
.hero-section,
.hero-section * ,
.abstract-section,
.abstract-section h3,
.abstract-section h1,
.abstract-section h2 {
  text-align: center !important;
}

/* Abstract 본문만 양쪽 정렬 */
.abstract-section .content {
  text-align: justify !important;
}

/* 데스크톱에서 기본 폰트 크게 */
@media screen and (min-width: 1024px) {
  body { overflow-x: hidden; }
  .narrow-container .content {
    font-size: 1.2rem;
    line-height: 1.9;
  }
  .figure-hero img {
    transform: scale(1.2);
    transform-origin: center;
    will-change: transform;
  }
}

/* 기본 이미지 반응형 */
.figure img {
  width: 100%;
  height: auto;
  display: block;
}

/* 모바일(≤768px): 이미지 중앙 */
@media screen and (max-width: 768px) {
  .figure img {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: auto;
    margin-right: auto;
  }
  .link-blocks {
    justify-content: center; /* 모바일에서는 버튼 중앙 */
  }
}
/* 모바일(≤768px): hero 버튼 크기 줄이기 */
@media screen and (max-width: 768px) {
  .hero-section .link-blocks .button.is-medium {
    font-size: 0.8rem;
    height: 2.2em;
    padding-left: 0.9em;
    padding-right: 0.9em;
    border-radius: 9999px;
  }
  .hero-section .link-blocks .button.is-medium .icon {
    font-size: 0.85em;
  }
}

/* 버튼 그룹 */
.link-blocks {
  display: flex;
  gap: .5rem;
  justify-content: flex-start;
  align-items: center;
}
.link-blocks .button + .button {
  margin-left: 0;
}

/* 이미지 여백 (기본값 유지) */
.section-figure {
  margin-top: 1rem;
  margin-bottom: 1.5rem;
}

/* 제목 크기 */
.h-title {
  font-size: clamp(1.75rem, 3.2vw, 2.75rem);
  font-weight: 900;
}
.h-subtitle {
  font-size: clamp(1.35rem, 2.4vw, 2.125rem);
  font-weight: 700;
}
.h-minor {
  font-size: clamp(1.2rem, 2vw, 1.625rem);
  font-weight: 700;
}

/* STEP 제목 강조 */
.step-title {
  font-size: clamp(2.25rem, 4.2vw, 3.25rem);
  font-weight: 900;
  letter-spacing: -0.01em;
  line-height: 1.15;
}

/* =========================
   Vertical Rhythm / Spacing
   ========================= */

/* 1) 간격 스케일 */
:root{
  --space-2xs: .4rem;
  --space-xs:  .8rem;
  --space-sm:  1.2rem;
  --space-md:  1.8rem;
  --space-lg:  2.4rem;
  --space-xl:  3.2rem;
  --space-2xl: 4.0rem;
}

/* 2) 섹션 여백: hero/abstract 는 원래대로, 그 외만 확대 */
.section:not(.hero-section):not(.abstract-section) {
  padding-top: var(--space-xl);
  padding-bottom: var(--space-xl);
}

/* 3) 같은 섹션 안에서 columns 묶음 사이 간격 키우기 (hero/abstract 제외 필요 없음) */
.section .columns + .columns {
  margin-top: var(--space-xl);
}

/* 4) 큰 제목/소제목 상하 간격 */
.h-title {
  margin-top: var(--space-xl);
  margin-bottom: var(--space-md);
}
.step-title {
  margin-top: var(--space-2xl);
  margin-bottom: var(--space-lg);
}
.h-subtitle {
  margin-top: var(--space-xl);
  margin-bottom: var(--space-sm);
}
.h-minor {
  margin-top: var(--space-lg);
  margin-bottom: var(--space-xs);
}

/* 5) 문단/표/목록 기본 간격 */
.content {
  margin-top: var(--space-sm);
  margin-bottom: var(--space-md);
}

/* 연속 블록 간 자동 간격 증대 */
:where(.content, .figure, .h-title, .h-subtitle, .h-minor)
  + :where(.content, .figure, .h-title, .h-subtitle, .h-minor) {
  margin-top: var(--space-lg);
}

/* 6) 이미지 블록 상하 여백 (확대) — hero/abstract 내부 기본 흐름은 유지 */
.section-figure {
  margin-top: var(--space-sm) !important;
  margin-bottom: var(--space-xl) !important;
}

/* 7) 리스트 간격 */
.content ul,
.content ol {
  margin-top: var(--space-xs);
  margin-bottom: var(--space-md);
}
.content li + li {
  margin-top: .35em;
}

/* 8) 히어로/링크 버튼 묶음 주변 여백 (기존 느낌 유지) */
.hero-section .figure-hero {
  margin-bottom: 1.5rem; /* 원래 감성 유지 */
}
.link-blocks {
  margin-top: var(--space-sm);
  margin-bottom: var(--space-md);
}

/* 9) 테이블/포스터 이미지 등 후속 블록 간격 */
.figure + .content,
.content + .figure {
  margin-top: var(--space-lg);
}

/* 10) 가독 좋은 구분선 */
hr.section-divider {
  margin: var(--space-xl) auto;
  border: none;
  border-top: 1px solid rgba(0,0,0,.12);
  width: min(980px, 100%);
}

/* 11) 모바일에서 여백 살짝 축소 */
@media screen and (max-width: 768px) {
  .section:not(.hero-section):not(.abstract-section) {
    padding-top: var(--space-lg);
    padding-bottom: var(--space-lg);
  }
  .h-title { margin-top: var(--space-lg); margin-bottom: var(--space-sm); }
  .step-title { margin-top: var(--space-xl); margin-bottom: var(--space-md); }
  .h-subtitle { margin-top: var(--space-lg); margin-bottom: var(--space-xs); }
  .h-minor { margin-top: var(--space-md); margin-bottom: var(--space-2xs); }
  .section .columns + .columns { margin-top: var(--space-lg); }
  .section-figure { margin-bottom: var(--space-lg) !important; }
  hr.section-divider { margin: var(--space-lg) auto; }
}
</style>

<!-- Hero Illustration + 링크 버튼 (그대로 유지) -->
<section class="section pt-4 pb-3 hero-section">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <div class="figure section-figure figure-hero">
          <img src="./static/image/intro.png" alt="Illustration">
        </div>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <div class="link-blocks mt-4" style="justify-content:center;">
          {% if page.paper_url %}
          <a href="{{ page.paper_url }}" target="_blank" rel="noopener"
             class="button is-dark is-rounded is-medium">
            <span class="icon"><i class="fas fa-file-pdf"></i></span><span>Paper</span>
          </a>
          {% endif %}
          <a href="./static/pdf/SurgX_Poster.pdf" target="_blank" rel="noopener"
             class="button is-dark is-rounded is-medium">
            <span class="icon"><i class="fas fa-file-pdf"></i></span><span>Poster</span>
          </a>
          {% if page.code_url %}
          <a href="{{ page.code_url }}" target="_blank" rel="noopener"
             class="button is-link is-rounded is-medium">
            <span class="icon"><i class="fab fa-github"></i></span><span>Code</span>
          </a>
          {% endif %}
        </div>
      </div>
    </div>
  </div>
</section>

<!-- Abstract (그대로 유지) -->
<section class="section pt-4 pb-4 abstract-section">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <h3 class="h-subtitle">Abstract</h3>
        <div class="content mt-3">
          Surgical phase recognition is central to workflow analysis, enabling applications such as monitoring, skill assessment, and process optimization. However, the underlying deep models are often black boxes, limiting interpretability and trust.
          <b>SurgX</b> is a concept-driven explanation framework that associates neurons with human-interpretable surgical concepts. We construct cholecystectomy-specific concept sets, identify representative neuron activation sequences, and annotate neurons with concepts.
          Evaluated on TeCNO and Causal ASFormer using Cholec80, SurgX yields meaningful explanations and improves transparency in surgical AI.
        </div>
      </div>
    </div>
  </div>
</section>

<hr class="section-divider">

<!-- Main Contributions (여기부터 시원한 간격 적용) -->
<section class="section pt-5 pb-5">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <h1 class="h-title">Main Contributions</h1>
        <ul class="content mt-4">
          <li>Introduce <strong>SurgX</strong>, a concept-based explanation framework for surgical phase recognition.</li>
          <li>Develop cholecystectomy-tailored concept sets and analyze best practices for concept selection.</li>
          <li>Validate SurgX on Causal ASFormer and TeCNO, showing consistent concept–neuron associations that enhance interpretability.</li>
        </ul>
      </div>
    </div>

    <hr class="section-divider">
    <div class="columns is-centered mt-6">
      <div class="column is-12-tablet is-10-desktop">
        <h1 class="h-title step-title">SurgX STEP 1. Neuron–Concept Annotation</h1>

        <div class="figure section-figure">
          <img src="./static/image/overall.png" alt="overall">
        </div>

        <div class="content">
          The pipeline for annotating concepts to neurons proceeds in three stages:
          <ol>
            <strong>A. Neuron Representative Sequence Selection</strong> – Select representative activation sequences for each neuron.
            <br>
            <strong>B. Concept Set Selection</strong> – Choose among three concept sets; <em>ChoLec-270</em> performs best in our study.
            <br>
            <strong>C. Neuron–Concept Association</strong> – Match neuron sequences with concepts via similarity in a surgical VLM space.
          </ol>
          Details of each stage are provided below.
        </div>

        <div class="content">
          <h3 class="h-subtitle" style="color:#3B6B1C;">A. Neuron Representative Sequence Selection</h3>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/representative-sequence-selection.png" alt="representative sequence selection">
        </div>

        <div class="content">
          Given a trained temporal phase recognizer (e.g., Causal ASFormer or TeCNO), we first select frames that yield high activations in the penultimate layer. Because temporal models respond to sequences rather than single frames, we extend each selected frame with its preceding frames to form a representative sequence. Ablation studies are summarized below.
        </div>

        <div class="content">
          <h4 class="h-minor" style="color:#3B6B1C;">Ablation Study: Frame Selection</h4>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/table2.png" alt="frame selection ablation">
        </div>

        <div class="content">
          <h4 class="h-minor" style="color:#3B6B1C;">Ablation Study: Sequence Length</h4>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/table3.png" alt="sequence length ablation">
        </div>

        <div class="content">
          <h3 class="h-subtitle" style="color:#5F2A96;">B. Concept Set Selection</h3>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/concept_set.png" alt="concept set selection">
        </div>

        <div class="content">
          Appropriate concept coverage is critical: if a neuron’s behavior is not representable by the concept set, reliable annotation is impossible. We therefore construct three cholecystectomy-related concept sets and compare them empirically.
        </div>

        <div class="content">
          <h4 class="h-minor" style="color:#5F2A96;">Ablation Study: Concept Sets</h4>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/table1.png" alt="concept set ablation">
        </div>

        <div class="content">
          <h3 class="h-subtitle" style="color:#4B8BAF;">C. Neuron–Concept Association</h3>
        </div>

        <div class="figure section-figure">
          <img src="./static/image/neuron-concept-association.png" alt="neuron–concept association">
        </div>

        <div class="content">
          Using the selected sequences and concept set, we compute cosine similarity in a surgical VLM space (e.g., SurgVLP, PeskaVLP) between each neuron’s representative sequence and each concept text, and assign to each neuron the concepts with highest similarity.
        </div>
      </div>
    </div>
    
    <hr class="section-divider">
    <div class="columns is-centered mt-6">
      <div class="column is-12-tablet is-10-desktop">
        <h1 class="h-title step-title">SurgX STEP 2. Model's Prediction Explanation</h1>

        <div class="figure section-figure">
          <img src="./static/image/models-prediction-explanation.png" alt="models prediction explanation">
        </div>

        <div class="content">
          The pipeline for annotating concepts to neurons proceeds in three stages:
          <ol>
            <li><strong>A. Neuron Representative Sequence Selection</strong> – Select representative activation sequences for each neuron.</li>
            <li><strong>B. Concept Set Selection</strong> – Choose among three concept sets; <em>ChoLec-270</em> performs best in our study.</li>
            <li><strong>C. Neuron–Concept Association</strong> – Match neuron sequences with concepts via similarity in a surgical VLM space.</li>
          </ol>
          Details of each stage are provided below.
        </div>
        <div class="content">
          <h4 class="h-subtitle">Qualitative Results – Explanation Examples</h4>
          <!-- === Video Player Block === -->
<div id="surgx-mp4-player" class="box" style="max-width: 980px; margin: 0 auto;">

  <!-- Video picker toolbar -->
  <div id="surgx-video-bar" class="buttons has-addons is-centered mb-3" style="justify-content:center;">
    <button class="button is-link is-light is-rounded is-small surgx-pick" data-name="video41">video41</button>
    <button class="button is-link is-light is-rounded is-small surgx-pick" data-name="video42">video42</button>
    <button class="button is-link is-light is-rounded is-small surgx-pick" data-name="video43">video43</button>
    <button class="button is-link is-light is-rounded is-small surgx-pick" data-name="video44">video44</button>
    <button class="button is-link is-light is-rounded is-small surgx-pick" data-name="video45">video45</button>
  </div>

  <!-- Video element -->
  <div style="display:flex; justify-content:center;">
    <video
      id="surgx-mp4"
      src="./static/video/video41.mp4"
      playsinline
      muted
      preload="metadata"
      style="max-width:100%; height:auto; background:#000;"
    ></video>
  </div>

  <!-- Controls -->
  <div class="mt-3" style="display:flex; align-items:center; gap:.75rem;">
    <button id="surgx-mp4-toggle" class="button is-dark is-rounded is-small">
      <span class="icon"><i class="fas fa-pause"></i></span>
      <span>Pause</span>
    </button>

    <input id="surgx-mp4-progress" type="range" min="1" max="1" value="1" step="1" style="flex:1;" />
    <span class="tag is-light is-rounded">
      <span id="surgx-mp4-cur">1</span>/<span id="surgx-mp4-total">1</span>
    </span>
  </div>

  <!-- Time display -->
  <div class="mt-2" style="display:flex; justify-content:flex-end; gap:.5rem; font-size:.9rem; color:#666;">
    <span><span id="surgx-time-cur">0:00</span> / <span id="surgx-time-total">0:00</span></span>
  </div>
</div>

<script>
(function() {
  // ===== 설정 =====
  const FPS = 2; // 프레임당 0.5초면 2

  const video    = document.getElementById('surgx-mp4');
  const btn      = document.getElementById('surgx-mp4-toggle');
  const progress = document.getElementById('surgx-mp4-progress');
  const curSpan  = document.getElementById('surgx-mp4-cur');
  const totSpan  = document.getElementById('surgx-mp4-total');
  const tCur     = document.getElementById('surgx-time-cur');
  const tTot     = document.getElementById('surgx-time-total');

  let totalFrames = 1;
  let rafId = null;
  let dragging = false;
  let rectCache = null;

  // ===== 유틸 =====
  const fmtTime = s => {
    if (!isFinite(s)) return '0:00';
    s = Math.max(0, Math.floor(s));
    const m = Math.floor(s/60), sec = s%60;
    return `${m}:${String(sec).padStart(2,'0')}`;
  };
  const frameToTime = f => (f - 1) / FPS;
  const curFrame = () => Math.min(totalFrames, Math.max(1, Math.round((video.currentTime||0)*FPS) + 1));
  const setFrame = f => {
    const clamped = Math.max(1, Math.min(totalFrames, f|0));
    const t = frameToTime(clamped);
    video.currentTime = t;
    progress.value = clamped;
    curSpan.textContent = String(clamped);
    tCur.textContent = fmtTime(t);
  };
  const posToFrame = (clientX) => {
    const rect = rectCache || progress.getBoundingClientRect();
    rectCache = rect;
    const x = (clientX - rect.left) / rect.width; // 0~1
    return Math.round(Math.max(0, Math.min(1, x)) * (totalFrames - 1)) + 1;
  };

  // ===== 메타데이터 로드 =====
  video.addEventListener('loadedmetadata', () => {
    const duration = video.duration || 0;
    totalFrames = Math.max(1, Math.round(duration * FPS));
    progress.max = totalFrames;
    progress.step = 1;
    progress.value = 1;
    curSpan.textContent = '1';
    totSpan.textContent = String(totalFrames);
    tTot.textContent = fmtTime(duration);
    video.play().catch(() => {
      btn.innerHTML = '<span class="icon"><i class="fas fa-play"></i></span><span>Play</span>';
    });
  });

  // ===== 재생/일시정지 =====
  video.addEventListener('play', () => {
    btn.innerHTML = '<span class="icon"><i class="fas fa-pause"></i></span><span>Pause</span>';
    if (!rafId) rafId = requestAnimationFrame(tick);
  });
  video.addEventListener('pause', () => {
    btn.innerHTML = '<span class="icon"><i class="fas fa-play"></i></span><span>Play</span>';
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    tick();
  });
  video.addEventListener('ended', () => {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    setFrame(totalFrames);
    btn.innerHTML = '<span class="icon"><i class="fas fa-play"></i></span><span>Play</span>';
  });
  btn.addEventListener('click', () => { if (video.paused) video.play(); else video.pause(); });

  // ===== 진행바 동기화 =====
  function tick() {
    const duration = video.duration || 0;
    if (duration > 0 && !dragging) {
      const f = curFrame();
      progress.value = f;
      curSpan.textContent = String(f);
    }
    tCur.textContent = fmtTime(video.currentTime || 0);
    rafId = requestAnimationFrame(tick);
  }

  // ===== 진행바 드래그/클릭 (1프레임 단위, pause 안 함) =====
  progress.addEventListener('pointerdown', (e) => {
    dragging = true; rectCache = null;
    progress.setPointerCapture(e.pointerId);
    e.preventDefault();
    setFrame(posToFrame(e.clientX));
  });
  progress.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    e.preventDefault();
    setFrame(posToFrame(e.clientX));
  });
  progress.addEventListener('pointerup', (e) => {
    dragging = false; rectCache = null;
    progress.releasePointerCapture(e.pointerId);
    e.preventDefault();
  });
  progress.addEventListener('input', () => { setFrame(parseInt(progress.value, 10) || 1); });
  progress.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { e.preventDefault(); setFrame(curFrame() + 1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); setFrame(curFrame() - 1); }
  });

  // ===== 비디오 선택 버튼 =====
  const BASE_PATH = './static/video/';
  const pickerBtns = document.querySelectorAll('.surgx-pick');
  function markActive(btn) {
    pickerBtns.forEach(b => b.classList.remove('is-link'));
    pickerBtns.forEach(b => b.classList.add('is-light'));
    btn.classList.remove('is-light');
    btn.classList.add('is-link');
  }
  function resetUIBeforeLoad() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    progress.disabled = true;
    progress.value = 1;
    curSpan.textContent = '1';
    totSpan.textContent = '…';
    tCur.textContent = '0:00';
    tTot.textContent = '0:00';
  }
  function loadVideoByName(name, autoplay=true) {
    resetUIBeforeLoad();
    video.pause();
    video.src = BASE_PATH + name + '.mp4';
    video.load();
    if (autoplay) video.play().catch(()=>{});
  }
  pickerBtns.forEach(btnEl => {
    btnEl.addEventListener('click', () => {
      markActive(btnEl);
      loadVideoByName(btnEl.dataset.name, true);
    });
  });
  // 초기 active 표시
  const current = (video.currentSrc || video.src || '').split('/').pop() || '';
  const matched = Array.from(pickerBtns).find(b => current.includes(b.dataset.name));
  if (matched) markActive(matched);
  else {
    const defBtn = document.querySelector('.surgx-pick[data-name="video41"]');
    if (defBtn) { markActive(defBtn); loadVideoByName('video41', true); }
  }
  video.addEventListener('loadedmetadata', () => { progress.disabled = false; });

})();
</script>

        </div>
      </div>
    </div>
  </div>
</section>



