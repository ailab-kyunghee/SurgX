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
/* --- Desktop 확대/모바일 풀폭 설정 --- */

/* 데스크톱(≥1216px): 본문 폭 살짝 확대 */
@media screen and (min-width: 1216px) {
  .narrow-container {
    max-width: 1200px;
    margin: 0 auto;
  }
}

/* 와이드스크린(≥1408px): 본문 폭 더 확대 */
@media screen and (min-width: 1408px) {
  .narrow-container {
    max-width: 1280px; /* 기존 1100 → 1280 */
    margin: 0 auto;
  }
}

/* 데스크톱에서 기본 폰트 크게 */
@media screen and (min-width: 1024px) {
  body { overflow-x: hidden; } /* 히어로 이미지 1.2배로 키워도 가로 스크롤 방지 */
  .narrow-container .content {
    font-size: 1.2rem;
    line-height: 1.9;
  }
  /* PC: 히어로(1.png)만 1.2배 */
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

/* 모바일(≤768px): 이미지 중앙 + 부모 폭 100%만 사용 */
@media screen and (max-width: 768px) {
  .figure {
    margin-left: 0;
    margin-right: 0;
  }
  .figure img {
    width: 100% !important;   /* 부모(컬럼) 너비만 사용 */
    max-width: 100% !important;
    display: block;
    margin-left: auto;
    margin-right: auto;       /* 확실한 중앙 정렬 */
  }
  .link-blocks .button.is-medium {
    font-size: 0.875rem;   /* 텍스트 크기 축소 */
    height: 2.25em;        /* 버튼 높이 축소 */
    padding-left: 1em;     /* 좌우 패딩 축소 */
    padding-right: 1em;
    border-radius: 9999px; /* pill 유지 */
  }
  /* 아이콘이 너무 크면 약간만 축소 */
  .link-blocks .button.is-medium .icon {
    font-size: 0.95em;
  }
  /* 버튼이 1줄에 꽉 차면 줄바꿈 허용(선택) */
  .link-blocks {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    justify-content: center;
  }
}

/* 버튼 그룹 간격 */
.link-blocks .button + .button {
  margin-left: .5rem;
}

/* 공용 이미지 여백 */
.section-figure {
  margin-top: 1rem;
  margin-bottom: 1.5rem;
}

/* 제목들: PC에서 더 크게 보이도록 상한 확대 */
.h-title { /* 메인 제목 */
  font-size: clamp(1.75rem, 3.2vw, 2.75rem);
  font-weight: 700;
}
.h-subtitle { /* 섹션 제목 */
  font-size: clamp(1.35rem, 2.4vw, 2.125rem);
  font-weight: 700;
}
.h-minor { /* 소제목 */
  font-size: clamp(1.2rem, 2vw, 1.625rem);
  font-weight: 700;
}
</style>

<!-- Hero Illustration + 링크 버튼 -->
<section class="section pt-4 pb-3">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <!-- 첫 이미지에 figure-hero 클래스 추가 -->
        <div class="figure section-figure figure-hero">
          <img src="./static/image/intro.png" alt="Illustration">
        </div>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <div class="link-blocks has-text-centered mt-4">
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
          <!-- {% if page.code_url %}
          <a href="https://ailab-kyunghee.github.io/SSG-Com/"
            class="button is-link is-rounded is-medium">
            <span class="icon"><i class="fas fa-database" aria-hidden="true"></i></span>
            <span>Dataset (Coming Sep 23)</span>
          </a>
          {% endif %} -->
        </div>
      </div>
    </div>

  </div>
</section>

<!-- Abstract -->
<section class="section pt-4 pb-4">
  <div class="container narrow-container">
    <div class="columns is-centered abstract-section">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h3 class="h-subtitle">Abstract</h3>
        <div class="content has-text-justified mt-3">
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
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h1 class="h-title">Main Contributions</h1>
        <ul class="content has-text-left mt-4" style="display:inline-block; text-align:left;">
          <li>Proposed <strong>SurgX</strong>, the first concept-based explanation framework for surgical phase recognition.</li>
          <li>Developed specialized concept sets for cholecystectomy videos and analyzed best practices for concept selection.</li>
          <li>Validated SurgX on two models(Causal ASFormer, TeCNO), demonstrating meaningful concept–neuron associations that enhance interpretability.</li>
        </ul>
      </div>
    </div>
    <div class="columns is-centered mt-6">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h2 class="h-title">SurgX, A Framework to Explain Surgical Phase Recognition Model</h2>
        <div class="figure section-figure">
          <img src="./static/image/neuron-concept-annotation.png" alt="concept set construction">
        </div>
        <div class="content has-text-justified">
          <h3 class="h-subtitle" style="color:#3B6B1C;">A. Neuron Representative Sequence Selection</h3>
        </div>
        <div class="content has-text-justified">
          <h3 class="h-subtitle" style="color:#3B6B1C;">B. Concept Set Selection</h3>
        </div>
        <div class="content has-text-justified">
          <h3 class="h-subtitle" style="color:#3B6B1C;">C. Neuron-Concept Association</h3>
        </div>
        <div class="figure section-figure">
          <img src="./static/image/models-prediction-explanation.png" alt="concept set construction">
        </div>
        <div class="content has-text-justified">
          <p>Details about concept set 1, 2, and 3 go here.</p>
        </div>
      </div>
    </div>
    <div class="columns is-centered mt-6">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h2 class="h-title">Concept Set Construction</h2>
        <div class="figure section-figure">
          <img src="./static/image/concept_set.png" alt="concept set construction">
        </div>
        <div class="content has-text-justified">
          <p>Details about concept set 1, 2, and 3 go here.</p>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- Dataset Comparison -->
<section class="section pt-4 pb-5">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h3 class="h-subtitle">Dataset Comparison</h3>
      </div>
    </div>
    <div class="columns is-centered mt-3">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <div class="figure section-figure">
          <img src="./static/image/3.png" alt="Dataset Comparison">
        </div>
        <div class="content has-text-justified">
          <p>This table contrasts the datasets used in previous surgical scene graph studies with Endoscapes-SG201.</p>
          <ul>
            <li>Endoscapes-SG201 is designed with holistic scene graph research in mind.</li>
            <li>It incorporates:
              <ul>
                <li>Diverse tools and anatomical structures as graph nodes.</li>
                <li>Diverse relationships as graph edges.</li>
                <li>Hand Identity labels as attributes of the tool nodes.</li>
              </ul>
            </li>
            <li>By unifying these elements, the dataset provides a more expressive and comprehensive foundation for modeling surgical scenes.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- Endoscapes-SG201 Details -->
<section class="section pt-4 pb-5">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h3 class="h-subtitle">Endoscapes-SG201 Details</h3>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <div class="figure section-figure">
          <img src="./static/image/4.png" alt="Endoscapes-SG201 Dataset Details">
        </div>
        <div class="content has-text-justified">
          <p>This table presents the category-wise distribution of the additional labels introduced in Endoscapes-SG201.</p>
          <p><b>Additional Annotations:</b></p>
          <ul>
            <li><b>6 Surgical Instruments</b>: Hook (HK), Grasper (GP), Clipper (CL), Bipolar (BP), Irrigator (IG), Scissors (SC)</li>
            <li><b>6 Surgical Actions</b>: Dissect (Dis.), Retract (Ret.), Grasp (Gr.), Clip (Cl.), Coagulate (Co.), Null</li>
            <li><b>3 Hand Identities</b>: Operator’s Right Hand (Rt), Operator’s Left Hand (Lt), Assistant’s Hand (Assi)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- SSG-Com -->
<section class="section pt-5 pb-5">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h2 class="h-title">SSG-Com</h2>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <div class="figure section-figure">
          <img src="./static/image/5.png" alt="SSG-Com Overall Architecture">
        </div>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <div class="content has-text-justified">
          <p><b>SSG-Com</b> is designed to leverage the diverse labels of Endoscapes-SG201.</p>
          <ol>
            <li>
              <b>Graph Construction</b><br>
              <b>Nodes</b>: Surgical instruments (with Hand identity), Anatomical structures<br>
              <b>Edges</b>: Spatial relations, Surgical action relations
            </li>
            <li class="mt-3">
              <b>Multi-task Training (3 classifiers)</b><br>
              <b>Classifier 1</b>: Spatial relation classification<br>
              <b>Classifier 2</b>: Action relation classification<br>
              <b>Classifier 3</b>: Hand identity classification
              <div class="mt-2 math-block">
                <b>Total Loss</b>:
                  \[
                  L_{\text{total}} = L_{\text{LG}} + \lambda_{\text{action}} L_{\text{action}} + \lambda_{\text{hand}} L_{\text{hand}} \tag*{}
                  \]
              </div>
            </li>
          </ol>
        </div>
      </div>
    </div>
  </div>
</section>

---

<!-- Experimental Results -->
<section class="section pt-5 pb-4">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h1 class="h-title">Experimental Results</h1>
      </div>
    </div>
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop">
        <div class="content mt-3">
          <p>The latent graph of SSG-Com demonstrated its effectiveness across two downstream tasks.</p>
          <ul>
            <li>Action Triplet Recognition</li>
            <li>CVS prediction</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="columns is-centered mt-4">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h2 class="h-subtitle">Quantitative Results</h2>
        <div class="figure section-figure">
          <img src="./static/image/6.png" alt="Quantitative Results">
        </div>
        <div class="content has-text-justified">
          <p><b>In Action Triplet Recognition (a):</b></p>
          <ul>
            <li>Modeling action relations as graph edges between nodes improved performance from 18.0 mAP (LG-CVS) to 23.5.</li>
            <li>Further incorporating Hand Identity increased performance to 24.2.</li>
          </ul>
          <p class="mt-3"><b>In CVS Prediction (b):</b></p>
          <ul>
            <li>Using Endoscapes-SG201 improved the performance of LG-CVS by 0.9 mAP, and SSG-Com achieved the highest score of 64.6.</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="columns is-centered mt-5">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <h2 class="h-subtitle">Qualitative Results</h2>
        <div class="figure section-figure">
          <img src="./static/image/7.png" alt="Qualitative Results">
        </div>
        <div class="content has-text-justified">
          By employing Endoscapes-SG201 and SSG-Com, we demonstrate the ability to construct a richer holistic surgical scene graph compared to existing approaches.
        </div>
      </div>
    </div>

  </div>
</section>

<!-- Collaborations -->
<section class="section pt-5 pb-6">
  <div class="container narrow-container">
    <div class="columns is-centered">
      <div class="column is-12-tablet is-10-desktop has-text-centered">
        <div class="figure section-figure">
          <img src="./static/image/8.png" alt="Collaborations">
        </div>
        <div class="content has-text-centered">
          <p class="is-size-6 has-text-grey mt-2">
            The authors thank Ms. Haeun Kim, M.F.A., for her professional assistance with the illustrations in this work.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>