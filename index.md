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
        Surgical phase recognition plays a crucial role in surgical workflow analysis, enabling applications such as monitoring, skill assessment, and workflow optimization. However, deep learning models remain black-boxes, limiting interpretability and trust. 
        <b>SurgX</b> is a novel concept-based explanation framework that associates neurons with human-interpretable surgical concepts. We construct concept sets tailored to cholecystectomy, select representative neuron activation sequences, and annotate neurons with concepts. 
        By evaluating on TeCNO and Causal ASFormer using Cholec80, we demonstrate that SurgX provides meaningful explanations and improves transparency in surgical AI.  
        </div>
      </div>
    </div>
  </div>
</section>

---

<!-- Main Contributions -->
<section class="section pt-5 pb-5">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <h1 class="h-title">Main Contributions</h1>
        <ul class="content mt-4">
          <li>Proposed <strong>SurgX</strong>, the first concept-based explanation framework for surgical phase recognition.</li>
          <li>Developed specialized concept sets for cholecystectomy videos and analyzed best practices for concept selection.</li>
          <li>Validated SurgX on two models (Causal ASFormer, TeCNO), demonstrating meaningful concept–neuron associations that enhance interpretability.</li>
        </ul>
      </div>
    </div>
    <div class="columns is-centered mt-6">
      <div class="column is-12-tablet is-10-desktop">
        <h1 class="h-title step-title">SurgX STEP 1. Neuron-Concept Annotation</h1>
        <div class="figure section-figure">
          <img src="./static/image/overall.png" alt="overall">
        </div>
        <div class="content">
          <h3 class="h-subtitle" style="color:#3B6B1C;">A. Neuron Representative Sequence Selection</h3>
        </div>
        <div class="figure section-figure">
          <img src="./static/image/representative-sequence-selection.png" alt="representative sequence selection">
        </div>
        <div class="figure section-figure">
          <img src="./static/image/table2.png" alt="representative sequence selection">
        </div>
        <div class="figure section-figure">
          <img src="./static/image/table3.png" alt="representative sequence selection">
        </div>
        <div class="content">
          <h3 class="h-subtitle" style="color:#5F2A96;">B. Concept Set Selection</h3>
        </div>
        <div class="figure section-figure">
          <img src="./static/image/concept_set.png" alt="concept set selection">
        </div>
        <div class="figure section-figure">
          <img src="./static/image/table1.png" alt="representative sequence selection">
        </div>
        <div class="content">
          <h3 class="h-subtitle" style="color:#4B8BAF;">C. Neuron-Concept Association</h3>
        </div>
        <div class="figure section-figure">
          <img src="./static/image/neuron-concept-association.png" alt="neuron-concept association">
        </div>
        <div class="content">
          <p>Details about concept set 1, 2, and 3 go here.</p>
        </div>
      </div>
    </div>
  </div>
</section>
