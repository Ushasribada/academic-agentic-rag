import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
import time
import re


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Academic AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Academic AI Assistant")

st.write(
    "Trust-Aware Agentic RAG for Academic Question Answering"
)

st.divider()


# =========================================================
# BASIC STATUS
# =========================================================

st.success(
    "✅ Streamlit application started successfully."
)


# =========================================================
# GEMINI API CONFIGURATION
# =========================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "Gemini API key is not configured in this terminal."
    )

    st.info(
        "Set GEMINI_API_KEY in PowerShell and restart Streamlit."
    )

    st.stop()


try:

    client = genai.Client(
        api_key=api_key
    )

except Exception as e:

    st.error(
        "Unable to initialize Gemini API."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner(
    "Loading embedding model for the first time..."
):

    embedding_model = load_embedding_model()


st.success(
    "✅ Embedding model loaded."
)


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {

    "what",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "from",
    "by",
    "this",
    "that",
    "these",
    "those",
    "how",
    "why",
    "which",
    "who",
    "when",
    "where",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "should",
    "has",
    "have",
    "had",
    "its",
    "their",
    "they",
    "them",
    "than",
    "into",
    "using",
    "used",
    "use",
    "about",
    "research",
    "paper",
    "study"
}


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text):

    if not text:
        return set()

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9-]+",
        text.lower()
    )

    keywords = {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }

    return keywords


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_into_sentences(text):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) >= 15:

            cleaned.append(
                sentence
            )

    return cleaned


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    sentence_database = []

    for page_index, page in enumerate(
        reader.pages
    ):

        page_number = page_index + 1

        try:

            page_text = page.extract_text()

        except Exception:

            page_text = ""

        if not page_text:
            continue

        sentences = split_into_sentences(
            page_text
        )

        for sentence_index, sentence in enumerate(
            sentences
        ):

            sentence_database.append({

                "text": sentence,

                "page": page_number,

                "sentence_index": sentence_index

            })

    return sentence_database


# =========================================================
# CONTEXT WINDOW CREATION
# =========================================================

def build_context_windows(
    sentence_database,
    window_size=1
):

    context_windows = []

    for i, item in enumerate(
        sentence_database
    ):

        start = max(
            0,
            i - window_size
        )

        end = min(
            len(sentence_database),
            i + window_size + 1
        )

        context_items = sentence_database[
            start:end
        ]

        context_text = " ".join(
            x["text"]
            for x in context_items
        )

        context_windows.append({

            "text": context_text,

            "page": item["page"],

            "center_sentence": item["text"],

            "center_index": i

        })

    return context_windows


# =========================================================
# FAISS INDEX CREATION
# =========================================================

def create_faiss_index(
    context_windows
):

    texts = [
        item["text"]
        for item in context_windows
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index, embeddings


# =========================================================
# QUERY EXPANSION
# =========================================================

def expand_query(question):

    q = question.lower()

    expanded_terms = []

    if any(
        word in q
        for word in [
            "objective",
            "aim",
            "goal",
            "purpose"
        ]
    ):

        expanded_terms.extend([
            "objective",
            "aim",
            "goal",
            "purpose",
            "research objective",
            "research aim",
            "research goal",
            "proposed work",
            "main objective",
            "primary objective",
            "problem statement",
            "problem statement and objectives"
        ])

    if any(
        word in q
        for word in [
            "method",
            "methodology",
            "approach",
            "architecture",
            "workflow",
            "implementation"
        ]
    ):

        expanded_terms.extend([
            "methodology",
            "method",
            "approach",
            "proposed system",
            "architecture",
            "workflow",
            "implementation"
        ])

    if any(
        word in q
        for word in [
            "dataset",
            "datasets",
            "data"
        ]
    ):

        expanded_terms.extend([
            "dataset",
            "datasets",
            "data",
            "training data",
            "test data",
            "experimental setup"
        ])

    if any(
        word in q
        for word in [
            "result",
            "results",
            "performance",
            "accuracy",
            "evaluation",
            "metric",
            "metrics"
        ]
    ):

        expanded_terms.extend([
            "results",
            "results and discussion",
            "performance",
            "evaluation",
            "accuracy",
            "metrics",
            "experimental results"
        ])

    if any(
        word in q
        for word in [
            "limitation",
            "limitations",
            "challenge",
            "challenges",
            "drawback",
            "drawbacks"
        ]
    ):

        expanded_terms.extend([
            "limitations",
            "challenges",
            "drawbacks",
            "limitations and future work"
        ])

    if any(
        word in q
        for word in [
            "future",
            "improve",
            "improvement"
        ]
    ):

        expanded_terms.extend([
            "future work",
            "future scope",
            "future research",
            "future directions",
            "improvements",
            "enhancement"
        ])

    all_terms = [
        question.strip()
    ]

    existing_terms = {
        x.lower()
        for x in all_terms
    }

    for term in expanded_terms:

        if term.lower() not in existing_terms:

            all_terms.append(
                term
            )

            existing_terms.add(
                term.lower()
            )

    return " ".join(
        all_terms
    )


# =========================================================
# SECTION DETECTION
# =========================================================

def detect_section_type(question):

    q = question.lower()

    if any(
        word in q
        for word in [
            "objective",
            "objectives",
            "aim",
            "goal",
            "purpose"
        ]
    ):

        return "objective"

    if any(
        word in q
        for word in [
            "method",
            "methodology",
            "approach",
            "architecture",
            "workflow",
            "implementation"
        ]
    ):

        return "methodology"

    if any(
        word in q
        for word in [
            "dataset",
            "datasets",
            "training data",
            "test data",
            "data"
        ]
    ):

        return "dataset"

    if any(
        word in q
        for word in [
            "result",
            "results",
            "performance",
            "accuracy",
            "evaluation",
            "metric",
            "metrics"
        ]
    ):

        return "results"

    if any(
        word in q
        for word in [
            "limitation",
            "limitations",
            "challenge",
            "challenges",
            "drawback",
            "drawbacks"
        ]
    ):

        return "limitations"

    if any(
        word in q
        for word in [
            "future",
            "future work",
            "future scope",
            "future research",
            "improve",
            "improvement"
        ]
    ):

        return "future"

    return None


# =========================================================
# SECTION HEADINGS
# =========================================================

SECTION_HEADINGS = {

    "objective": [

        "problem statement and objectives",

        "research objectives",

        "objectives",

        "objective",

        "aim",

        "purpose"

    ],

    "methodology": [

        "methodology",

        "proposed methodology",

        "method",

        "methods",

        "approach",

        "proposed system",

        "system architecture",

        "architecture",

        "implementation"

    ],

    "dataset": [

        "datasets and experimental setup",

        "dataset and experimental setup",

        "datasets",

        "dataset",

        "data",

        "experimental setup"

    ],

    "results": [

        "results and discussion",

        "experimental results",

        "results",

        "evaluation",

        "performance evaluation",

        "performance"

    ],

    "limitations": [

        "limitations",

        "limitations and future work",

        "challenges",

        "drawbacks"

    ],

    "future": [

        "future work",

        "future scope",

        "future research",

        "future directions"

    ]

}


# =========================================================
# NORMAL RETRIEVAL
# =========================================================

def retrieve_evidence(
    question,
    context_windows,
    index,
    top_k=8
):

    if not context_windows:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    search_k = min(
        len(context_windows),
        max(top_k * 5, 20)
    )

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    question_keywords = extract_keywords(
        question
    )

    candidates = []

    for semantic_score, index_value in zip(
        scores[0],
        indices[0]
    ):

        if index_value < 0:
            continue

        item = context_windows[
            index_value
        ]

        text_keywords = extract_keywords(
            item["text"]
        )

        if (
            question_keywords
            and text_keywords
        ):

            common_keywords = (
                question_keywords
                .intersection(
                    text_keywords
                )
            )

            keyword_score = (
                len(common_keywords)
                / len(question_keywords)
            )

        else:

            keyword_score = 0.0

        semantic_score = float(
            max(
                0.0,
                min(
                    1.0,
                    semantic_score
                )
            )
        )

        combined_score = (
            0.75 * semantic_score
            + 0.25 * keyword_score
        )

        candidates.append({

            "text": item["text"],

            "page": item["page"],

            "center_sentence": item[
                "center_sentence"
            ],

            "semantic_score": semantic_score,

            "keyword_score": float(
                keyword_score
            ),

            "combined_score": float(
                combined_score
            ),

            "retrieval_type": "Hybrid"

        })

    candidates.sort(
        key=lambda item: item[
            "combined_score"
        ],
        reverse=True
    )

    selected = []

    seen = set()

    for item in candidates:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower().strip()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        selected.append(
            item
        )

        if len(selected) >= top_k:
            break

    return selected


# =========================================================
# SECTION-AWARE RETRIEVAL
# =========================================================

def section_aware_retrieve(
    question,
    sentence_database,
    top_k=5
):

    section_type = detect_section_type(
        question
    )

    if section_type is None:
        return []

    headings = SECTION_HEADINGS.get(
        section_type,
        []
    )

    if not headings:
        return []

    heading_indices = []

    for i, sentence_item in enumerate(
        sentence_database
    ):

        sentence_text = (
            sentence_item["text"]
            .lower()
            .strip()
        )

        for heading in headings:

            if heading.lower() in sentence_text:

                heading_indices.append(
                    i
                )

                break

    if not heading_indices:
        return []

    question_keywords = extract_keywords(
        question
    )

    candidates = []

    for heading_index in heading_indices:

        heading_item = sentence_database[
            heading_index
        ]

        page_number = heading_item[
            "page"
        ]

        heading_text = heading_item[
            "text"
        ].strip()

        evidence_units = []

        heading_lower_original = (
            heading_text.lower()
        )

        post_heading_text = ""

        for heading in headings:

            heading_lower = heading.lower()

            position = (
                heading_lower_original.find(
                    heading_lower
                )
            )

            if position != -1:

                post_heading_text = (
                    heading_text[
                        position
                        + len(heading):
                    ].strip()
                )

                break

        if post_heading_text:

            evidence_units.append(
                post_heading_text
            )

        start_index = (
            heading_index + 1
        )

        end_index = min(
            len(sentence_database),
            start_index + 6
        )

        for j in range(
            start_index,
            end_index
        ):

            item = sentence_database[
                j
            ]

            if item["page"] != page_number:
                break

            text = item["text"].strip()

            if not text:
                continue

            lower_text = text.lower()

            new_section_found = False

            for (
                other_section,
                other_headings
            ) in SECTION_HEADINGS.items():

                if other_section == section_type:
                    continue

                for other_heading in (
                    other_headings
                ):

                    if (
                        other_heading.lower()
                        in lower_text
                    ):

                        new_section_found = True

                        break

                if new_section_found:
                    break

            if new_section_found:
                break

            evidence_units.append(
                text
            )

        expanded_units = []

        for unit in evidence_units:

            unit = unit.strip()

            if not unit:
                continue

            if "•" in unit:

                bullet_parts = re.split(
                    r"\s*[•●▪◦]\s*",
                    unit
                )

                for part in bullet_parts:

                    part = part.strip()

                    if len(part) >= 15:

                        expanded_units.append(
                            part
                        )

            else:

                expanded_units.append(
                    unit
                )

        for evidence_text in expanded_units:

            text_keywords = extract_keywords(
                evidence_text
            )

            if (
                question_keywords
                and text_keywords
            ):

                common_keywords = (
                    question_keywords
                    .intersection(
                        text_keywords
                    )
                )

                keyword_score = (
                    len(common_keywords)
                    / len(question_keywords)
                )

            else:

                keyword_score = 0.0

            section_score = 0.70

            keyword_bonus = (
                0.30 * keyword_score
            )

            combined_score = (
                section_score
                + keyword_bonus
            )

            candidates.append({

                "text": evidence_text,

                "page": page_number,

                "center_sentence": heading_text,

                "semantic_score": 0.0,

                "keyword_score": float(
                    keyword_score
                ),

                "combined_score": float(
                    combined_score
                ),

                "retrieval_type": "Section-Aware"

            })

    selected = []

    seen = set()

    for item in candidates:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower().strip()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        selected.append(
            item
        )

    selected.sort(
        key=lambda item: item[
            "combined_score"
        ],
        reverse=True
    )

    return selected[:top_k]


# =========================================================
# COMBINE RETRIEVAL RESULTS
# =========================================================

def combine_evidence(
    normal_evidence,
    section_evidence,
    top_k=8
):

    combined = []

    seen = set()

    for item in section_evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower().strip()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        combined.append(
            item
        )

    for item in normal_evidence:

        normalized = re.sub(
            r"\s+",
            " ",
            item["text"].lower().strip()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        combined.append(
            item
        )

    combined.sort(
        key=lambda item: item[
            "combined_score"
        ],
        reverse=True
    )

    return combined[:top_k]


# =========================================================
# EVIDENCE VERIFICATION
# =========================================================

def verify_evidence(
    question,
    evidence
):

    if not evidence:

        return {

            "status": "Insufficient",

            "score": 0.0,

            "average_score": 0.0

        }

    best_score = max(
        item["combined_score"]
        for item in evidence
    )

    average_score = (
        sum(
            item["combined_score"]
            for item in evidence
        )
        / len(evidence)
    )

    if best_score >= 0.42:

        status = "Supported"

    elif best_score >= 0.32:

        status = "Partially Supported"

    else:

        status = "Insufficient"

    return {

        "status": status,

        "score": float(
            best_score
        ),

        "average_score": float(
            average_score
        )

    }


# =========================================================
# AGENTIC RE-RETRIEVAL
# =========================================================

def re_retrieve_evidence(
    question,
    sentence_database,
    context_windows,
    index,
    top_k=8
):

    expanded_query = expand_query(
        question
    )

    normal_evidence = retrieve_evidence(
        expanded_query,
        context_windows,
        index,
        top_k=top_k
    )

    section_evidence = (
        section_aware_retrieve(
            expanded_query,
            sentence_database,
            top_k=5
        )
    )

    final_evidence = combine_evidence(
        normal_evidence,
        section_evidence,
        top_k=top_k
    )

    return final_evidence


# =========================================================
# GEMINI MODEL CALL
# =========================================================

def generate_answer(
    question,
    evidence
):

    if not evidence:
        return None, None

    evidence_text = ""

    for i, item in enumerate(
        evidence,
        start=1
    ):

        evidence_text += (
            f"\nEvidence {i} "
            f"[Page {item['page']}]:\n"
            f"{item['text']}\n"
        )

    prompt = f"""
You are a trustworthy academic question-answering assistant.

Answer the user's question ONLY using the
provided evidence.

Do not use outside knowledge.

If the evidence is insufficient, clearly say
that the available document evidence is
insufficient.

Every factual statement must be supported by
the provided evidence.

Include page citations in this exact format:

[Page 1]

[Page 2]

Do not invent page numbers.

User Question:

{question}

Provided Evidence:

{evidence_text}

Write a clear and concise academic answer.
"""

    model_names = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash"
    ]

    last_error = None

    for model_name in model_names:

        try:

            start_time = time.time()

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            latency = (
                time.time()
                - start_time
            )

            answer = getattr(
                response,
                "text",
                None
            )

            if answer:

                return (

                    answer.strip(),

                    {

                        "model": model_name,

                        "latency": latency

                    }

                )

        except Exception as e:

            last_error = e

    return None, {

        "error": str(last_error)
        if last_error
        else "Unknown Gemini error"

    }


# =========================================================
# CITATION CORRECTNESS
# =========================================================

def check_citation_correctness(
    answer,
    evidence
):

    if not answer or not evidence:

        return {

            "score": 0.0,

            "status": "Unsupported",

            "citation_pages": [],

            "valid_pages": [],

            "invalid_pages": []

        }

    cited_pages = []

    page_patterns = [

        r"\[Page\s+(\d+)\]",

        r"\bPage\s+(\d+)\b"

    ]

    for pattern in page_patterns:

        matches = re.findall(
            pattern,
            answer,
            flags=re.IGNORECASE
        )

        for match in matches:

            page_number = int(
                match
            )

            if page_number not in cited_pages:

                cited_pages.append(
                    page_number
                )

    evidence_pages = sorted(
        set(
            item["page"]
            for item in evidence
        )
    )

    valid_pages = [

        page

        for page in cited_pages

        if page in evidence_pages

    ]

    invalid_pages = [

        page

        for page in cited_pages

        if page not in evidence_pages

    ]

    answer_sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip()
    )

    answer_sentences = [

        sentence.strip()

        for sentence in answer_sentences

        if len(sentence.strip()) >= 15

    ]

    evidence_texts = [

        item["text"]

        for item in evidence

    ]

    evidence_keyword_sets = [

        extract_keywords(text)

        for text in evidence_texts

    ]

    sentence_scores = []

    for sentence in answer_sentences:

        sentence_keywords = extract_keywords(
            sentence
        )

        if not sentence_keywords:
            continue

        best_overlap = 0.0

        for evidence_keywords in (
            evidence_keyword_sets
        ):

            if not evidence_keywords:
                continue

            overlap = (

                len(

                    sentence_keywords.intersection(
                        evidence_keywords
                    )

                )

                / len(sentence_keywords)

            )

            best_overlap = max(
                best_overlap,
                overlap
            )

        sentence_scores.append(
            best_overlap
        )

    if sentence_scores:

        support_score = (

            sum(sentence_scores)

            / len(sentence_scores)

        )

    else:

        support_score = 0.0

    if cited_pages:

        page_score = (

            len(valid_pages)

            / len(cited_pages)

        )

    else:

        page_score = 0.0

    final_score = (

        0.70 * support_score

        + 0.30 * page_score

    )

    if (

        final_score >= 0.70

        and cited_pages

        and not invalid_pages

    ):

        status = "Correctly Supported"

    elif final_score >= 0.45:

        status = "Partially Supported"

    else:

        status = "Weakly Supported"

    return {

        "score": float(
            final_score
        ),

        "status": status,

        "citation_pages": cited_pages,

        "valid_pages": valid_pages,

        "invalid_pages": invalid_pages

    }


# =========================================================
# DISPLAY EVIDENCE
# =========================================================

def display_evidence(
    evidence
):

    if not evidence:

        st.warning(
            "No relevant evidence found."
        )

        return

    for i, item in enumerate(
        evidence,
        start=1
    ):

        with st.expander(
            f"Evidence {i} — Page {item['page']}"
        ):

            st.write(
                item["text"]
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Semantic",
                    f"{item['semantic_score']:.4f}"
                )

            with col2:

                st.metric(
                    "Keyword",
                    f"{item['keyword_score']:.4f}"
                )

            with col3:

                st.metric(
                    "Combined",
                    f"{item['combined_score']:.4f}"
                )


# =========================================================
# SESSION STATE
# =========================================================

if "processed_file_name" not in st.session_state:

    st.session_state.processed_file_name = None


if "sentence_database" not in st.session_state:

    st.session_state.sentence_database = []


if "context_windows" not in st.session_state:

    st.session_state.context_windows = []


if "faiss_index" not in st.session_state:

    st.session_state.faiss_index = None


# =========================================================
# PDF UPLOAD
# =========================================================

st.subheader(
    "📄 Upload Academic Document"
)

uploaded_file = st.file_uploader(
    "Upload a PDF research paper or academic document",
    type=["pdf"]
)


if uploaded_file is not None:

    if (
        st.session_state.processed_file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF and building retrieval index..."
        ):

            sentence_database = process_pdf(
                uploaded_file
            )

            context_windows = build_context_windows(
                sentence_database,
                window_size=1
            )

            if not context_windows:

                st.error(
                    "No readable text was found in the uploaded PDF."
                )

                st.stop()

            faiss_index, _ = create_faiss_index(
                context_windows
            )

            st.session_state.sentence_database = (
                sentence_database
            )

            st.session_state.context_windows = (
                context_windows
            )

            st.session_state.faiss_index = (
                faiss_index
            )

            st.session_state.processed_file_name = (
                uploaded_file.name
            )

        st.success(
            f"✅ PDF processed successfully: "
            f"{len(sentence_database)} evidence sentences found."
        )

    else:

        st.success(
            f"✅ Using processed document: "
            f"{uploaded_file.name}"
        )


# =========================================================
# QUESTION INPUT
# =========================================================

if st.session_state.faiss_index is not None:

    st.subheader(
        "💬 Ask a Question"
    )

    question = st.text_input(
        "Enter your academic question",
        placeholder=(
            "Example: What is the main objective of this research?"
        )
    )

    ask_button = st.button(
        "🔎 Ask Academic AI",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

            st.stop()

        sentence_database = (
            st.session_state.sentence_database
        )

        context_windows = (
            st.session_state.context_windows
        )

        faiss_index = (
            st.session_state.faiss_index
        )


        # =================================================
        # INITIAL RETRIEVAL
        # =================================================

        with st.spinner(
            "Retrieving relevant academic evidence..."
        ):

            normal_evidence = retrieve_evidence(
                question,
                context_windows,
                faiss_index,
                top_k=8
            )

            section_evidence = section_aware_retrieve(
                question,
                sentence_database,
                top_k=5
            )

            evidence = combine_evidence(
                normal_evidence,
                section_evidence,
                top_k=8
            )

            verification = verify_evidence(
                question,
                evidence
            )

        retrieval_stage = (
            "Initial Retrieval"
        )


        # =================================================
        # AGENTIC RE-RETRIEVAL
        # =================================================

        if verification["status"] in [

            "Partially Supported",

            "Insufficient"

        ]:

            st.warning(
                "⚠️ Initial evidence was not strong enough. "
                "Agent is performing a second retrieval."
            )

            with st.spinner(
                "Agent is expanding the query and "
                "re-retrieving evidence..."
            ):

                expanded_query = expand_query(
                    question
                )

                evidence = re_retrieve_evidence(
                    question,
                    sentence_database,
                    context_windows,
                    faiss_index,
                    top_k=8
                )

                verification = verify_evidence(
                    expanded_query,
                    evidence
                )

            retrieval_stage = (
                "Agentic Re-Retrieval"
            )

            with st.expander(
                "🔄 Agentic Query Expansion"
            ):

                st.write(
                    expanded_query
                )


        # =================================================
        # EVIDENCE STATUS
        # =================================================

        st.subheader(
            "📊 Evidence Verification"
        )

        status_col1, status_col2, status_col3 = (
            st.columns(3)
        )

        with status_col1:

            st.metric(
                "Evidence Status",
                verification["status"]
            )

        with status_col2:

            st.metric(
                "Retrieval Stage",
                retrieval_stage
            )

        with status_col3:

            st.metric(
                "Best Evidence Relevance",
                f"{verification['score']:.4f}"
            )

        if evidence:

            st.caption(
                f"Average evidence relevance: "
                f"{verification['average_score']:.4f}"
            )


        # =================================================
        # RETRIEVED RELEVANT CONTENT
        # =================================================

        st.subheader(
            "📚 Retrieved Relevant Content"
        )

        display_evidence(
            evidence
        )


        # =================================================
        # GEMINI ANSWER
        # =================================================

        st.subheader(
            "🤖 AI Answer"
        )

        with st.spinner(
            "Generating a source-grounded academic answer..."
        ):

            answer, model_info = generate_answer(
                question,
                evidence
            )

        if answer:

            st.write(
                answer
            )

            if model_info:

                st.caption(
                    f"Model: "
                    f"{model_info.get('model', 'Unknown')} | "
                    f"Latency: "
                    f"{model_info.get('latency', 0.0):.2f} seconds"
                )


            # =================================================
            # SUPPORTING EVIDENCE
            # =================================================

            st.subheader(
                "🔎 Supporting Evidence"
            )

            for i, item in enumerate(
                evidence,
                start=1
            ):

                st.markdown(
                    f"**Evidence {i} — Page {item['page']}**"
                )

                st.write(
                    item["text"]
                )


            # =================================================
            # CITATION CORRECTNESS
            # =================================================

            citation_result = (
                check_citation_correctness(
                    answer,
                    evidence
                )
            )

            st.subheader(
                "🧾 Citation Correctness"
            )

            citation_col1, citation_col2 = (
                st.columns(2)
            )

            with citation_col1:

                st.metric(
                    "Citation Status",
                    citation_result["status"]
                )

            with citation_col2:

                st.metric(
                    "Citation Support Score",
                    f"{citation_result['score']:.2f}"
                )

            if citation_result[
                "citation_pages"
            ]:

                st.write(
                    "**Cited pages:** "
                    + ", ".join(
                        str(page)
                        for page in citation_result[
                            "citation_pages"
                        ]
                    )
                )

                st.write(
                    "**Valid cited pages:** "
                    + ", ".join(
                        str(page)
                        for page in citation_result[
                            "valid_pages"
                        ]
                    )
                )

                if citation_result[
                    "invalid_pages"
                ]:

                    st.warning(
                        "Invalid cited pages: "
                        + ", ".join(
                            str(page)
                            for page in citation_result[
                                "invalid_pages"
                            ]
                        )
                    )

            else:

                st.info(
                    "No page citation was detected "
                    "in the generated answer."
                )

        else:

            error_message = (
                "The AI answer could not be generated."
            )

            if (
                model_info
                and model_info.get("error")
            ):

                error_message = (
                    "The AI answer could not be generated. "
                    "Gemini error: "
                    + model_info["error"]
                )

            st.error(
                error_message
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Trust-Aware Agentic RAG for Academic Question Answering"
)
