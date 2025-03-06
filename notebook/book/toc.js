// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="intro.html">Introduction</a></li><li class="chapter-item expanded "><a href="current.html"><strong aria-hidden="true">1.</strong> Current Focus</a></li><li class="chapter-item expanded "><div><strong aria-hidden="true">2.</strong> Recommender Systems</div></li><li><ol class="section"><li class="chapter-item expanded "><a href="recsys/gradient_boosting.html"><strong aria-hidden="true">2.1.</strong> Gradient Boosting</a></li><li class="chapter-item expanded "><a href="recsys/tfidf.html"><strong aria-hidden="true">2.2.</strong> TF-IDF</a></li><li class="chapter-item expanded "><a href="recsys/cross_encoders.html"><strong aria-hidden="true">2.3.</strong> Cross Encoders</a></li><li class="chapter-item expanded "><a href="recsys/sentence_transformers.html"><strong aria-hidden="true">2.4.</strong> SentenceTransformers</a></li><li class="chapter-item expanded "><a href="recsys/collab_filtering.html"><strong aria-hidden="true">2.5.</strong> Collaborative Filtering</a></li><li class="chapter-item expanded "><a href="recsys/evaluation.html"><strong aria-hidden="true">2.6.</strong> Evaluation</a></li></ol></li><li class="chapter-item expanded "><a href="ab_test/init.html"><strong aria-hidden="true">3.</strong> AB Testing</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="ab_test/examples.html"><strong aria-hidden="true">3.1.</strong> Examples</a></li><li class="chapter-item expanded "><a href="ab_test/power_analysis.html"><strong aria-hidden="true">3.2.</strong> Power Analysis</a></li></ol></li><li class="chapter-item expanded "><a href="llm/llm.html"><strong aria-hidden="true">4.</strong> LLMs</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="llm/fine_tuning.html"><strong aria-hidden="true">4.1.</strong> Fine-tuning</a></li><li class="chapter-item expanded "><a href="llm/useful_models.html"><strong aria-hidden="true">4.2.</strong> Useful Models</a></li><li class="chapter-item expanded "><a href="llm/encoder_vs_decoder.html"><strong aria-hidden="true">4.3.</strong> Encoder vs Decoder</a></li><li class="chapter-item expanded "><a href="llm/contextualized_recs.html"><strong aria-hidden="true">4.4.</strong> Contextualized Recommendations</a></li></ol></li><li class="chapter-item expanded "><a href="misc.html"><strong aria-hidden="true">5.</strong> Miscellaneous</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="misc/bradley-terry.html"><strong aria-hidden="true">5.1.</strong> Bradley-Terry Model</a></li><li class="chapter-item expanded "><a href="misc/wsl-setup.html"><strong aria-hidden="true">5.2.</strong> Setting up WSL</a></li><li class="chapter-item expanded "><a href="misc/to-read.html"><strong aria-hidden="true">5.3.</strong> To Read</a></li><li class="chapter-item expanded "><a href="misc/packages.html"><strong aria-hidden="true">5.4.</strong> Packages</a></li><li class="chapter-item expanded "><a href="misc/skills.html"><strong aria-hidden="true">5.5.</strong> Skills</a></li><li class="chapter-item expanded "><a href="misc/hash_collision.html"><strong aria-hidden="true">5.6.</strong> Hash Collisions</a></li></ol></li><li class="chapter-item expanded "><a href="identities.html"><strong aria-hidden="true">6.</strong> Identities</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="identities/sigmoid.html"><strong aria-hidden="true">6.1.</strong> Sigmoid</a></li><li class="chapter-item expanded "><a href="identities/statistics.html"><strong aria-hidden="true">6.2.</strong> Statistics</a></li></ol></li><li class="chapter-item expanded "><a href="papers.html"><strong aria-hidden="true">7.</strong> Papers</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="papers/weinberger_2009.html"><strong aria-hidden="true">7.1.</strong> Weinberger 2009 - Hashing for Multitask Learning</a></li><li class="chapter-item expanded "><a href="papers/rendle_2009.html"><strong aria-hidden="true">7.2.</strong> Rendle 2009 - Bayesian Personalized Ranking</a></li><li class="chapter-item expanded "><a href="papers/burges_2010.html"><strong aria-hidden="true">7.3.</strong> Burges 2010 - RankNET to LambdaMART</a></li><li class="chapter-item expanded "><a href="papers/schroff_2015.html"><strong aria-hidden="true">7.4.</strong> Schroff 2015 - FaceNET</a></li><li class="chapter-item expanded "><a href="papers/covington_2016.html"><strong aria-hidden="true">7.5.</strong> Covington 2016 - Deep NNs for Youtube Recs</a></li><li class="chapter-item expanded "><a href="papers/schnabel_2016.html"><strong aria-hidden="true">7.6.</strong> Schnabel 2016 - Recs as Treatments</a></li><li class="chapter-item expanded "><a href="papers/bateni_2017.html"><strong aria-hidden="true">7.7.</strong> Bateni 2017 - Affinity Clustering</a></li><li class="chapter-item expanded "><a href="papers/guo_2017.html"><strong aria-hidden="true">7.8.</strong> Guo 2017 - DeepFM</a></li><li class="chapter-item expanded "><a href="papers/hamilton_2017.html"><strong aria-hidden="true">7.9.</strong> Hamilton 2017 - GraphSAGE</a></li><li class="chapter-item expanded "><a href="papers/ma_2018.html"><strong aria-hidden="true">7.10.</strong> Ma 2018 - Entire Space Multi-Task Model</a></li><li class="chapter-item expanded "><a href="papers/kang_2018.html"><strong aria-hidden="true">7.11.</strong> Kang 2018 - SASRec</a></li><li class="chapter-item expanded "><a href="papers/reimers_2019.html"><strong aria-hidden="true">7.12.</strong> Reimers 2019 - Sentence-BERT</a></li><li class="chapter-item expanded "><a href="papers/yi_2019.html"><strong aria-hidden="true">7.13.</strong> Yi 2019 - LogQ Correction for In Batch Sampling</a></li><li class="chapter-item expanded "><a href="papers/zhao_2019.html"><strong aria-hidden="true">7.14.</strong> Zhao 2019 - Recommending What to Watch Next</a></li><li class="chapter-item expanded "><a href="papers/lee_2020.html"><strong aria-hidden="true">7.15.</strong> Lee 2020 - Large Scale Video Representation Learning</a></li><li class="chapter-item expanded "><a href="papers/he_2020.html"><strong aria-hidden="true">7.16.</strong> He 2020 - LightGCN</a></li><li class="chapter-item expanded "><a href="papers/lewis_2020.html"><strong aria-hidden="true">7.17.</strong> Lewis 2020 - Retrieval Augmented Generation</a></li><li class="chapter-item expanded "><a href="papers/gao_2021.html"><strong aria-hidden="true">7.18.</strong> Gao 2021 - SimCSE</a></li><li class="chapter-item expanded "><a href="papers/weng_2021.html"><strong aria-hidden="true">7.19.</strong> Weng 2021 - Contrastive Representation Learning</a></li><li class="chapter-item expanded "><a href="papers/dao_2022.html"><strong aria-hidden="true">7.20.</strong> Dao 2022 - Flash Attention</a></li><li class="chapter-item expanded "><a href="papers/li_2021.html"><strong aria-hidden="true">7.21.</strong> Li 2021 - TaoBao Embedding-Based Retrieval</a></li><li class="chapter-item expanded "><a href="papers/zou_2021.html"><strong aria-hidden="true">7.22.</strong> Zou 2021 - PLM Based Ranking in Baidu Search</a></li><li class="chapter-item expanded "><a href="papers/tunstall_2022.html"><strong aria-hidden="true">7.23.</strong> Tunstall 2022 - SetFit</a></li><li class="chapter-item expanded "><a href="papers/rafailov_2023.html"><strong aria-hidden="true">7.24.</strong> Rafailov 2023 - Direct Preference Optimization</a></li><li class="chapter-item expanded "><a href="papers/blecher_2023.html"><strong aria-hidden="true">7.25.</strong> Blecher 2023 - Nougat</a></li><li class="chapter-item expanded "><a href="papers/dong_2023.html"><strong aria-hidden="true">7.26.</strong> Dong 2023 - MINE Loss</a></li><li class="chapter-item expanded "><a href="papers/liu_2023.html"><strong aria-hidden="true">7.27.</strong> Liu 2023 - Meaning Representations from Trajectories</a></li><li class="chapter-item expanded "><a href="papers/klenitskiy_2023.html"><strong aria-hidden="true">7.28.</strong> Klenitskiy 2023 - BERT4Rec vs SASRec</a></li><li class="chapter-item expanded "><a href="papers/singh_2023.html"><strong aria-hidden="true">7.29.</strong> Singh 2023 - Semantic IDs for Recs</a></li><li class="chapter-item expanded "><a href="papers/borisyuk_2024.html"><strong aria-hidden="true">7.30.</strong> Borisyuk 2024 - GNN at LinkedIn</a></li><li class="chapter-item expanded "><a href="papers/wang_2024.html"><strong aria-hidden="true">7.31.</strong> Wang 2024 - LLM for Pinterest Search</a></li></ol></li><li class="chapter-item expanded "><a href="nlp_course/intro.html"><strong aria-hidden="true">8.</strong> NLP Course</a></li><li class="chapter-item expanded "><a href="database_course/intro.html"><strong aria-hidden="true">9.</strong> Database Course</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="database_course/lecture01.html"><strong aria-hidden="true">9.1.</strong> Lecture 1</a></li><li class="chapter-item expanded "><a href="database_course/lecture02.html"><strong aria-hidden="true">9.2.</strong> Lecture 2</a></li><li class="chapter-item expanded "><a href="database_course/lecture03.html"><strong aria-hidden="true">9.3.</strong> Lecture 3</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
