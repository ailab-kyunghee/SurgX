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
    font-size: 0.8rem;     /* 글자 크기 줄이기 */
    height: 2.2em;         /* 버튼 높이 축소 */
    padding-left: 0.9em;   /* 좌우 패딩 축소 */
    padding-right: 0.9em;
    border-radius: 9999px; /* pill 모양 유지 */
  }

  .hero-section .link-blocks .button.is-medium .icon {
    font-size: 0.85em; /* 아이콘 살짝 축소 */
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

/* 이미지 여백 */
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

/* 2) 섹션 자체 여백 강화 (Bulma pt/pb 클래스를 보완) */
.section {
  padding-top: var(--space-xl);
  padding-bottom: var(--space-xl);
}

/* 3) 같은 섹션 안에서 columns 묶음 사이 간격 키우기 */
.section .columns + .columns {
  margin-top: var(--space-xl);
}

/* 4) 큰 제목(챕터)과 소제목(서브챕터)의 상하 간격 */
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

/* 5) 문단(.content)과 목록, 표 등의 기본 간격 */
.content {
  margin-top: var(--space-sm);
  margin-bottom: var(--space-md);
}

/* 연속된 블록들 사이 간격 자동 증대 */
:where(.content, .figure, .h-title, .h-subtitle, .h-minor) 
  + :where(.content, .figure, .h-title, .h-subtitle, .h-minor) {
  margin-top: var(--space-lg);
}

/* 6) 이미지 블록 상하 여백 통일 */
.section-figure {
  margin-top: var(--space-sm) !important;
  margin-bottom: var(--space-xl) !important;
}

/* 7) 리스트/목록 간격 */
.content ul,
.content ol {
  margin-top: var(--space-xs);
  margin-bottom: var(--space-md);
}
.content li + li {
  margin-top: .35em; /* 항목 간 살짝 띄움 */
}

/* 8) 히어로/링크 버튼 묶음 주변 여백 */
.hero-section .figure-hero {
  margin-bottom: var(--space-lg);
}
.link-blocks {
  margin-top: var(--space-sm);
  margin-bottom: var(--space-md);
}

/* 9) 테이블/포스터 이미지 등 후속 블록 간격 통일감 */
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

/* 11) 모바일에서 과도한 여백 축소 (가독 유지) */
@media screen and (max-width: 768px) {
  .section {
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

<!-- Hero Illustration + 링크 버튼 -->
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

<!-- Abstract -->
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

<!-- Main Contributions -->
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
            <li><strong>A. Neuron Representative Sequence Selection</strong> – Select representative activation sequences for each neuron.</li>
            <li><strong>B. Concept Set Selection</strong> – Choose among three concept sets; <em>ChoLec-270</em> performs best in our study.</li>
            <li><strong>C. Neuron–Concept Association</strong> – Match neuron sequences with concepts via similarity in a surgical VLM space.</li>
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
  </div>
</section>
