import hashlib
import os
import inspect
import json
import logging
import re
import shelve
from datetime import datetime, timezone, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple
from threading import Lock

import torch
import torch.nn.functional as F
import ollama
from app.services.guardrails import (
    contains_prompt_injection_attempt,
    prompt_injection_response,
)
from app.services.contact_store import get_contact, upsert_contact

DEVICE = torch.device("cpu")
WIB = timezone(timedelta(hours=7))

# =========================
# ANSI colors
# =========================
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# =========================
# Path setup
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".cache"
EMBED_CACHE_DIR = DATA_DIR / "embeddings"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Defaults / Config
# =========================
DEFAULT_VAULT = str(DATA_DIR / "vault.txt")
THREADS_DB_PATH = str(PROJECT_ROOT / "threads_db_store" / "threads_db")
DEFAULT_EMBED_MODEL = "mxbai-embed-large:latest"
DEFAULT_GEN_MODEL = "deepseek-r1:latest"

DEFAULT_TOP_K = int("20")
DEFAULT_GEN_TOP_P = float("0.9")
DEFAULT_MIN_SIMILARITY = float("0.55")
DEFAULT_GEN_TEMPERATURE = float("0.15")
DEFAULT_GEN_REPEAT_PENALTY = float("1.20")

DEFAULT_NUM_CTX = int("16384")
DEFAULT_MIN_MSG_LENGTH = int("40")
DEFAULT_MAX_MSG_LENGTH = int("2048")
DEFAULT_MAX_HISTORY_TURNS = int("12")
DEFAULT_MODEL_HISTORY_WINDOW = int("5")
DEFAULT_MAX_SEEN_IDS = int(os.getenv("WHATSAPP_MAX_SEEN_IDS", "500"))
ALLOWED_TOPIC_PATTERNS = [
    re.compile(r"\boptimaxx\b", re.IGNORECASE),
    re.compile(r"\bansys\b", re.IGNORECASE),
    re.compile(r"\bio[\s-]?t\b", re.IGNORECASE),
    re.compile(r"\binternet[\s-]+of[\s-]+things\b", re.IGNORECASE),
    re.compile(r"\bcae\b", re.IGNORECASE),
]
HELP_INTENT_ACADEMIC = "ACADEMIC"
HELP_INTENT_BUSINESS = "BUSINESS"
HELP_INTENT_OTHER = "OTHER"

OPTIMAXX_SYSTEM_PROMPT = dedent(
    """
    Anda adalah asisten chatbot Optimaxx (software rekayasa & layanan IoT). Selalu jawab dalam bahasa Indonesia dengan gaya karyawan Optimaxx: ramah, profesional, jelas, singkat. Ikuti aturan:

    KEAMANAN DATA:
    - Jangan berikan info sensitif/rahasia perusahaan.
    - Jangan menebak/mengarang info karyawan, pelanggan, atau sistem internal.
    - Jika ditanya hal sensitif: jawab "Maaf, saya tidak dapat membagikan informasi internal atau pribadi terkait karyawan atau data perusahaan."

    ATURAN JAWABAN:
    - Maksimum ~450 kata.
    - Jawab secara general saja, jangan menjawab terlalu spesifik kecuali ada di context.
    - Jika pertanyaan terlalu rumit, minta customer untuk kontak tim Optimaxx.
    - Optimaxx hanya melayani industri besar.
    - Optimaxx tidak memiliki produk hardware untuk konsumen, hanya solusi pemasangan hardware skala industri.
    - Jika context tidak mengandung jawaban untuk pertanyaan, jangan membuat jawaban sendiri. Jawab mohon maaf dan sarankan menghubungi tim Optimaxx.
    - Jangan membuat asumsi atau informasi yang tidak disebutkan.
    - Jangan membuat informasi baru di luar context.
    - Jangan berikan arah rute ke kantor, hanya alamat.
    - Gunakan emoji sewajarnya.

    LARANGAN AKADEMIK:
    Optimaxx tidak membantu tugas akademik apa pun, termasuk:
    - Skripsi, tesis, disertasi
    - Tugas kuliah, tugas akhir, UTS/UAS
    - Makalah, laporan, proposal akademik
    - Pengerjaan tugas coding atau joki
    - Revisi skripsi atau pembuatan BAB
    - Presentasi kampus atau rangkuman teori
    - Les privat atau mentoring akademik

    Jika terdeteksi pertanyaan akademik/student-related, jawab WAJIB:
    “Optimaxx saat ini lebih berfokus pada layanan konsultasi dan dukungan teknis untuk industri besar, bukan untuk kebutuhan akademik atau skripsi. 🙂”

    ❗PENTING:
    - **Hanya dalam kasus pertanyaan akademik/student-related saja**, tambahkan rekomendasi YouTube:
      - Ajak menonton YouTube Optimaxx: https://www.youtube.com/@optimaxxprimateknik6957
      - Jelaskan manfaat Member (video eksklusif, rekaman pelatihan, konten teknis tambahan).
    - **Untuk pertanyaan non-akademik (bisnis/teknis/engineering)** → JANGAN merekomendasikan YouTube.

    ❗ ATURAN TRAINING / KELAS / PRIVATE CLASS:
    Jika pengguna bertanya tentang:
    - Training / kelas / kursus
    - Private class
    - Pelatihan ANSYS/CAE/IoT
    - Jadwal atau biaya kelas
    - Training in-house/online/offline

    Maka jawab WAJIB:
    “Optimaxx tidak menyediakan layanan training (kelas online/offline maupun private). Namun jika ingin belajar, Anda bisa menggunakan konten di YouTube Optimaxx atau bergabung sebagai YouTube Member 🙂”

    (YouTube boleh direkomendasikan di sini karena konteksnya adalah permintaan belajar/training, bukan bisnis.)

    SUMBER INFORMASI:
    - Hanya boleh menggunakan informasi yang tertulis eksplisit di context.
    - Jika context kosong dan topiknya tidak ada di sistem:
      - Jangan mengarang data, pengalaman, atau angka.
      - Jawab WAJIB:
        "Maaf, berdasarkan informasi yang ada di sistem kami, saat ini belum ada data atau contoh spesifik yang bisa kami bagikan terkait pertanyaan tersebut. Untuk informasi lebih lanjut, silakan menghubungi tim Optimaxx ya. 🙂"
      - Ajak hubungi tim Optimaxx, tanpa menambahkan detail baru.

    ATURAN ANGKA & KONTAK:
    - Tidak boleh mengarang angka apa pun: nomor telepon, WhatsApp, harga, biaya, tarif, persentase, jumlah klien, tahun, durasi, kuota, dll.
    - Hanya boleh menggunakan kontak yang tertulis eksplisit.
    - Jika pengguna menanyakan nomor/harga/kontak yang tidak ada:
      - Jangan membuat nomor contoh atau placeholder.
      - Arahkan ke website resmi Optimaxx.

    KONTAK (selalu di akhir jawaban):
    📧 Email: info@optimaxx.id
    🌐 Website: https://optimaxx.id
    ▶️ YouTube: https://www.youtube.com/@optimaxxprimateknik6957
    🏢 Alamat: The Mansion Bougenville, Tower Fontana Unit BF33E2, Jl. Trembesi Blok D, Kemayoran - Jakarta Utara 14410
    """
).strip()

HUMAN_HANDOFF_RESPONSE = dedent(
    """
    Baik, kami akan segera menghubungkan Anda dengan tim Optimaxx.  
    Mohon tunggu sebentar ya—tim kami sedang dalam perjalanan untuk membantu Anda. 😊  

    Jika dalam beberapa menit belum ada balasan, Anda bisa menghubungi kami melalui:  
    📧 Email: info@optimaxx.id  
    🌐 Website: https://optimaxx.id  
    🏢 Alamat: The Mansion Bougenville, Tower Fontana Unit BF33E2, Jl. Trembesi Blok D, Kemayoran - Jakarta Utara 14410
    """
).strip()

ACADEMIC_REFUSAL_MESSAGE = dedent(
    """
    Optimaxx berfokus pada layanan konsultasi dan dukungan teknis untuk industri besar, sehingga kami tidak dapat membantu kebutuhan akademik seperti skripsi, tugas kuliah, atau laporan.

    Untuk belajar mandiri, silakan kunjungi YouTube Optimaxx: 
    https://www.youtube.com/@optimaxxprimateknik6957

    YouTube Member menyediakan video eksklusif, rekaman pelatihan, dan konten teknis tambahan.

    📧 Email: info@optimaxx.id
    🌐 Website: https://optimaxx.id
    """
).strip()

# =========================
# Utils
# =========================
def strip_think(text: str) -> str:
    """
    Remove any think blocks from the text.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    """
    Try to parse a JSON object from raw LLM output.
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def safe_print_info(msg: str) -> None:
    print(NEON_GREEN + msg + RESET_COLOR)


def safe_print_warn(msg: str) -> None:
    print(YELLOW + msg + RESET_COLOR)


def safe_print_section(title: str) -> None:
    print(PINK + title + RESET_COLOR)


# =========================
# Embeddings + Retrieval
# =========================
def embed_texts_ollama(texts: List[str], model: str) -> List[List[float]]:
    """
    Embed a list of texts with Ollama. Returns list of embedding vectors.
    """
    if not texts:
        return []

    client_factory = getattr(ollama, "Client", None)
    client = None
    if callable(client_factory):
        try:
            client = client_factory()
        except Exception as err:
            safe_print_warn(f"[warn] gagal membuat Ollama client: {err}")
            client = None

    def _embed_once(prompt: str) -> List[float] | None:
        try:
            if client is not None:
                response = client.embeddings(model=model, prompt=prompt)
            else:
                response = ollama.embeddings(model=model, prompt=prompt)
            return list(response["embedding"])
        except Exception as exc:
            safe_print_warn(f"[warn] embedding failed for a line: {exc}")
            return None

    cache: Dict[str, List[float]] = {}
    vectors: List[List[float]] = []
    for text in texts:
        key = text.strip()
        if key in cache:
            vectors.append(list(cache[key]))
            continue
        vec = _embed_once(text)
        if vec is None:
            safe_print_warn("[warn] embedding skipped due to previous error.")
            continue
        cache[key] = vec
        vectors.append(list(vec))

    if len(vectors) != len(texts):
        safe_print_warn(
            "[warn] embedding count mismatch; skipping partial embeddings to avoid misalignment."
        )
        return []
    return vectors


def cosine_topk(
    query_vec: torch.Tensor,
    doc_matrix: torch.Tensor,
    k: int,
    *,
    doc_matrix_normalized: bool = False,
) -> List[Tuple[int, float]]:
    """
    Cosine similarity between single query [D] and doc_matrix [N, D].
    Returns list of (index, score) pairs sorted by similarity.
    """
    if doc_matrix.numel() == 0:
        return []
    if doc_matrix.device != query_vec.device:
        doc_matrix = doc_matrix.to(query_vec.device)
    q = F.normalize(query_vec.unsqueeze(0), dim=1)  # [1, D]
    d = (
        doc_matrix if doc_matrix_normalized else F.normalize(doc_matrix, dim=1)
    )  # [N, D]
    sims = torch.mm(q, d.transpose(0, 1)).squeeze(0)  # [N]
    k = min(k, sims.numel())
    if k == 0:
        return []
    values, indices = torch.topk(sims, k=k)
    return [(indices[i].item(), values[i].item()) for i in range(k)]


def get_relevant_context(
    rewritten_input: str,
    vault_embeddings: torch.Tensor,
    vault_content: List[str],
    top_k: int,
    embed_model: str,
    min_similarity: float,
    *,
    vault_embeddings_norm: torch.Tensor | None = None,
) -> List[str]:
    """
    Embed the query, cosine-match against vault embeddings, return top chunks above threshold.
    """

    if rewritten_input.strip() == "":
        safe_print_warn(f"Input is empty; skipping retrieval.")
        return []
    if vault_embeddings.numel() == 0:
        return []
    try:
        q_emb = ollama.embeddings(model=embed_model, prompt=rewritten_input)[
            "embedding"
        ]
    except Exception as e:
        safe_print_warn(f"[warn] embeddings failed via Ollama: {e}")
        return []
    q_vec = torch.tensor(q_emb, dtype=torch.float32, device=DEVICE)  # [D]
    search_matrix = (
        vault_embeddings_norm if vault_embeddings_norm is not None else vault_embeddings
    )
    top_hits = cosine_topk(
        q_vec,
        search_matrix,
        top_k,
        doc_matrix_normalized=vault_embeddings_norm is not None,
    )
    return [
        vault_content[idx].strip() for idx, score in top_hits if score >= min_similarity
    ]


# =========================
# Generation (Ollama)
# =========================
def ollama_chat_call(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = DEFAULT_GEN_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    top_p: float | None = None,
    top_k: int | None = None,
    repeat_penalty: float | None = None,
) -> str:
    """
    messages = [{"role":"system","content":"..."}, {"role":"user","content":"..."} ...]
    """
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    resp = ollama.chat(model=model, messages=messages, options=options)
    return strip_think(resp["message"]["content"])


def build_guardrail_system_message(fallback: str | None) -> str:
    """
    Build the system message for guardrail checks.
    """
    return fallback or OPTIMAXX_SYSTEM_PROMPT


def _contains_allowed_topic(text: str) -> bool:
    """
    Check if text contains any allowed topic keywords.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in ALLOWED_TOPIC_PATTERNS)


def _collect_user_history_text(
    messages: List[Dict[str, Any]], *, limit: int = DEFAULT_MODEL_HISTORY_WINDOW
) -> str:
    """
    Join recent user-authored, AI-readable messages for classifier context.
    """
    user_lines: List[str] = []
    for msg in messages:
        if not msg.get("ai_readable", True):
            continue
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            user_lines.append(content)
    if limit > 0:
        user_lines = user_lines[-limit:]
    return "\n".join(user_lines).strip()


def _run_combined_checks(
    question: str,
    user_history_text: str,
    model: str = DEFAULT_GEN_MODEL,
) -> Dict[str, Any]:
    """
    Run one lightweight classifier call that returns multiple guard flags.
    Output schema (all keys required):
    {
      "needs_human": bool,
      "handoff_reason": str,
      "help_intent": "ACADEMIC"|"BUSINESS"|"OTHER",
      "is_domain_question": bool,
      "can_answer_without_context": bool
    }
    """
    defaults = {
        "needs_human": False,
        "handoff_reason": "",
        "help_intent": HELP_INTENT_OTHER,
        "is_domain_question": True,
        "can_answer_without_context": False,
    }

    if not question:
        return defaults

    instruction = dedent(
        """
        Kamu adalah pengklasifikasi cepat untuk pesan WhatsApp Optimaxx.
        Balas HANYA dengan satu JSON sesuai skema:
        {
          "needs_human": true|false,
          "handoff_reason": "alasan singkat dalam bahasa Indonesia",
          "help_intent": "ACADEMIC"|"BUSINESS"|"OTHER",
          "is_domain_question": true|false,
          "can_answer_without_context": true|false
        }

        Aturan:
        - needs_human panduan:
            - Jawab true HANYA jika:
            1) Pengirim JELAS dan EKPLISIT meminta untuk berbicara dengan manusia, agen, admin, atau dukungan manual
                (misalnya: "mau bicara dengan admin", "tolong hubungkan ke orang", "saya mau live agent"),
                
            DAN

            2) Isi pesan menunjukkan bahwa kasusnya PENTING atau TIDAK COCOK ditangani otomatis,
                misalnya: masalah pembayaran/transaksi, akses akun/keamanan, kendala teknis berulang
                setelah beberapa kali coba, atau eskalasi komplain serius.

            - Jika pesan hanya berisi pertanyaan umum, salam, komplain ringan, atau permintaan bantuan yang masih bisa dijawab oleh bot (misalnya FAQ, penjelasan produk, cara pakai software), jawablah false, bahkan jika pengguna menyebut kata "bantu".
            - Jika pengguna meminta manusia/admin secara umum tetapi konteksnya tampak sederhana atau tidak jelas penting/krisis, utamakan menjawab false.
            - Jawab false untuk semua maksud lain (pertanyaan umum, permintaan info, keluhan tanpa permintaan manusia yang jelas dan penting, dsb.).
            - Jika kamu RAGU apakah benar-benar perlu intervensi manusia, SELALU pilih false.
            
        - handoff_reason: Jelaskan secara singkat dalam bahasa Indonesia mengapa perlu intervensi manusia, atau kosongkan jika needs_human = false.
        - help_intent: ACADEMIC jika terkait skripsi / tugas kuliah / dosen / pelajaran / training akademik / joki; BUSINESS jika soal layanan Optimaxx / ANSYS / IoT / CAE / kerja sama; lainnya = OTHER.
        - is_domain_question = true jika topik dalam lingkup Optimaxx / ANSYS / IoT / CAE, selain itu false.
        - can_answer_without_context = true hanya jika pertanyaan bisa dijawab umum tanpa dokumen tambahan; kalau ragu pilih false.
        - Jangan sertakan backtick, teks tambahan, atau penjelasan di luar objek JSON tersebut.
        """
    ).strip()

    user_payload = dedent(
        f"""
        Pesan terbaru: {question}
        Riwayat pengguna terkait: {user_history_text or "(tidak ada)"}
        """
    ).strip()

    try:
        raw = ollama_chat_call(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            num_ctx=1536,
            top_p=0.05,
            repeat_penalty=1.01,
        )
    except Exception as exc:
        logging.warning("Combined classifier failed: %s", exc)
        return defaults

    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        logging.warning("Combined classifier returned non-dict: %s", raw)
        return defaults

    out = dict(defaults)
    out["needs_human"] = bool(parsed.get("needs_human", defaults["needs_human"]))
    out["handoff_reason"] = str(parsed.get("handoff_reason") or "").strip()
    help_intent = str(parsed.get("help_intent") or "").strip().upper()
    if help_intent in {HELP_INTENT_ACADEMIC, HELP_INTENT_BUSINESS, HELP_INTENT_OTHER}:
        out["help_intent"] = help_intent
    out["is_domain_question"] = bool(
        parsed.get("is_domain_question", defaults["is_domain_question"])
    )
    out["can_answer_without_context"] = bool(
        parsed.get("can_answer_without_context", defaults["can_answer_without_context"])
    )
    return out


def chat_with_rag(
    user_input: str,
    system_message: str | None,
    vault_embeddings: torch.Tensor,
    vault_content: List[str],
    ollama_model: str,
    embed_model: str,
    conversation_history: List[Dict[str, Any]],
    user_entry: Dict[str, Any],
    *,
    vault_embeddings_norm: torch.Tensor | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = DEFAULT_GEN_TEMPERATURE,
    top_p: float = DEFAULT_GEN_TOP_P,
    repeat_penalty: float = DEFAULT_GEN_REPEAT_PENALTY,
    precomputed_checks: Dict[str, Any] | None = None,
) -> str:
    """
    Main single-turn RAG call: rewrite -> retrieve -> respond with context, history, or fallback.
    """
    conversation_history.append(user_entry)
    rewritten = user_input
    logging.debug("Incoming question: %s", user_input)

    embeddings_norm = vault_embeddings_norm
    if embeddings_norm is None:
        embeddings_norm = _VAULT_EMB_NORM
    if embeddings_norm is None and vault_embeddings.numel() > 0:
        embeddings_norm = _normalize_embedding_matrix(vault_embeddings)

    filtered_history = [
        msg for msg in conversation_history if msg.get("ai_readable", True)
    ]
    previous_messages = filtered_history[:-1]
    if DEFAULT_MAX_HISTORY_TURNS <= 1 or DEFAULT_MODEL_HISTORY_WINDOW <= 0:
        previous_messages = []
    else:
        history_window = min(
            DEFAULT_MODEL_HISTORY_WINDOW, DEFAULT_MAX_HISTORY_TURNS - 1
        )
        previous_messages = (
            previous_messages[-history_window:] if history_window else []
        )

    history_lines: List[str] = []
    user_history_lines: List[str] = []
    for msg in previous_messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if role and content:
            history_lines.append(f"{role}: {content}")
        elif role:
            history_lines.append(f"{role}:")
        elif content:
            history_lines.append(content)
        if role.lower() == "user" and content:
            user_history_lines.append(content)
    history_text = "\n".join(history_lines).strip()
    history_available = bool(history_text)
    default_history_text = (
        history_text
        if history_available
        else "Tidak ada riwayat percakapan sebelumnya."
    )
    user_history_text = "\n".join(user_history_lines).strip()

    checks = precomputed_checks or {}
    help_intent = checks.get("help_intent", HELP_INTENT_OTHER)
    if conversation_history:
        conversation_history[-1]["help_intent"] = help_intent
    logging.debug("Help intent classification: %s", help_intent)
    if help_intent == HELP_INTENT_ACADEMIC:
        if conversation_history and conversation_history[-1].get("role") == "user":
            conversation_history[-1]["ai_readable"] = False
        conversation_history.append(
            _make_message("system", ACADEMIC_REFUSAL_MESSAGE, ai_readable=False)
        )
        return ACADEMIC_REFUSAL_MESSAGE

    is_domain_question = checks.get("is_domain_question")
    if is_domain_question is None:
        is_domain_question = bool(
            _contains_allowed_topic(rewritten) or _contains_allowed_topic(user_history_text)
        )
    else:
        is_domain_question = bool(is_domain_question)
    logging.debug("Is domain question: %s", is_domain_question)
    if not is_domain_question:
        safe_print_warn(
            "Pertanyaan berada di luar cakupan Optimaxx/ANSYS/IoT/CAE. Mengirim penolakan."
        )
        out_of_scope_message = dedent(
            """
            Maaf, saya hanya dapat membantu pertanyaan yang berkaitan dengan layanan Optimaxx, ANSYS, IoT, atau CAE. 
            Silakan ajukan pertanyaan dalam ruang lingkup tersebut agar saya bisa membantu dengan tepat. 🙂

            📧 Email: info@optimaxx.id
            🌐 Website: https://optimaxx.id
            """
        ).strip()
        if conversation_history and conversation_history[-1].get("role") == "user":
            conversation_history[-1]["ai_readable"] = False
        conversation_history.append(
            _make_message("system", out_of_scope_message, ai_readable=False)
        )
        return out_of_scope_message

    context_chunks = get_relevant_context(
        rewritten_input=rewritten,
        vault_embeddings=vault_embeddings,
        vault_content=vault_content,
        top_k=top_k,
        embed_model=embed_model,
        min_similarity=min_similarity,
        vault_embeddings_norm=embeddings_norm,
    )

    base_system = build_guardrail_system_message(system_message)

    if context_chunks:
        ctx = "\n".join(context_chunks)
        sys_content = (
            f"{base_system}\n\n"
            f"Konteks yang tersedia:\n{ctx}\n\n"
            f"Riwayat percakapan sebelumnya:\n{default_history_text}"
        )
    else:
        safe_print_info(CYAN + "Tidak ditemukan konteks yang relevan." + RESET_COLOR)
        can_answer_without_context = bool(
            checks.get("can_answer_without_context", False)
        )
        logging.debug("Can answer without context: %s", can_answer_without_context)
        if not can_answer_without_context:
            safe_print_warn("Melewati balasan karena tidak ada konteks relevan.")
            fallback_message = dedent(
                """
                Mohon maaf, saya belum memiliki informasi mengenai hal tersebut saat ini. 
                Untuk bantuan lebih lanjut, silakan menghubungi tim Optimaxx ya. 😊

                📧 Email: info@optimaxx.id
                🌐 Website: https://optimaxx.id
                """
            ).strip()
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history[-1]["ai_readable"] = False
            conversation_history.append(
                _make_message("system", fallback_message, ai_readable=False)
            )
            return fallback_message

        general_prompt_parts = [
            base_system,
            "Tidak ada konteks dokumen yang relevan saat ini.",
        ]

        if history_available:
            general_prompt_parts.append(
                f"Riwayat percakapan sebelumnya:\n{history_text}"
            )
        else:
            general_prompt_parts.append("Riwayat percakapan sebelumnya tidak tersedia.")
        general_prompt_parts.append(
            dedent(
                """
                Instruksi tambahan:
                - Jawab hanya jika pertanyaan dapat dijelaskan secara umum atau berdasarkan riwayat yang diberikan.
                - Jika saat menjawab Anda menyadari bahwa informasi spesifik masih hilang, katakan dengan sopan bahwa data tersebut belum tersedia dan sarankan pengguna menghubungi tim Optimaxx.
                """
            ).strip()
        )
        sys_content = "\n\n".join(general_prompt_parts)

    out = ollama_chat_call(
        model=ollama_model,
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": rewritten},
        ],
        temperature=temperature,
        num_ctx=num_ctx,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )

    conversation_history.append(_make_message("assistant", out, ai_readable=True))
    if (
        DEFAULT_MAX_HISTORY_TURNS > 0
        and len(conversation_history) > DEFAULT_MAX_HISTORY_TURNS
    ):
        del conversation_history[:-DEFAULT_MAX_HISTORY_TURNS]
    return out


# =========================
# Vault helpers
# =========================
def load_vault(path: str | Path) -> List[str]:
    """
    Read vault file where each non-empty line is one chunk.
    """
    vault_path = Path(path)
    if not vault_path.is_file():
        safe_print_warn(
            f"[info] vault file not found at {vault_path}; retrieval will be empty."
        )
        return []
    try:
        with vault_path.open("r", encoding="utf-8") as f:
            chunks = [line.strip() for line in f if line.strip()]
    except OSError as exc:
        safe_print_warn(f"[warn] failed to read vault file {vault_path}: {exc}")
        return []
    safe_print_info(f"Loaded {len(chunks)} vault entries from {vault_path}.")
    return chunks


def build_vault_embeddings(
    chunks: List[str], embed_model: str, vault_path: str | Path
) -> torch.Tensor:
    """
    Embed each chunk into a [N, D] tensor (or empty tensor if none), reusing cache when possible.
    """
    if not chunks:
        return torch.empty((0,), dtype=torch.float32, device=DEVICE)
    vault_path = Path(vault_path)
    try:
        abs_vault_path = vault_path.resolve(strict=False)
    except OSError:
        abs_vault_path = vault_path
    sanitized_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", embed_model)
    cache_prefix = f"{abs_vault_path.name}.{sanitized_model}"
    cache_tensor_path = EMBED_CACHE_DIR / f"{cache_prefix}.pt"
    cache_meta_path = EMBED_CACHE_DIR / f"{cache_prefix}.json"
    hasher = hashlib.sha256()
    hasher.update(str(embed_model).encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(str(abs_vault_path).encode("utf-8"))
    hasher.update(b"\n")
    for chunk in chunks:
        hasher.update(chunk.encode("utf-8"))
        hasher.update(b"\n")
    content_hash = hasher.hexdigest()

    if cache_tensor_path.exists() and cache_meta_path.exists():
        try:
            with cache_meta_path.open("r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
            if metadata.get("hash") == content_hash:
                safe_print_info("Memuat embedding vault dari cache...")
                load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
                try:
                    if "weights_only" in inspect.signature(torch.load).parameters:
                        load_kwargs["weights_only"] = True
                except Exception:
                    pass
                cached_tensor = torch.load(str(cache_tensor_path), **load_kwargs)
                if isinstance(cached_tensor, torch.Tensor):
                    return cached_tensor.to(DEVICE)
        except Exception as e:
            safe_print_warn(f"[warn] gagal memuat cache embedding: {e}")

    safe_print_info("Menghasilkan embedding untuk konten vault...")
    vectors = embed_texts_ollama(chunks, embed_model)
    if not vectors:
        safe_print_warn("[warn] tidak ada embedding yang valid; vault retrieval dinonaktifkan.")
        return torch.empty((0,), dtype=torch.float32, device=DEVICE)
    try:
        mat_cpu = torch.tensor(vectors, dtype=torch.float32)  # [N, D]
    except Exception as e:
        safe_print_warn(f"[warn] failed to create embeddings tensor: {e}")
        return torch.empty((0,), dtype=torch.float32, device=DEVICE)
    try:
        torch.save(mat_cpu, str(cache_tensor_path))
        with cache_meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump({"hash": content_hash, "shape": list(mat_cpu.shape)}, meta_file)
        safe_print_info("Embedding vault baru berhasil disimpan ke cache.")
    except Exception as e:
        safe_print_warn(f"[warn] gagal menyimpan cache embedding: {e}")
    safe_print_info(f"Bentuk embedding: {list(mat_cpu.shape)}")
    return mat_cpu.to(DEVICE)


def _normalize_embedding_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Return a row-wise normalized copy to avoid repeated cosine normalization at query time."""
    if not isinstance(matrix, torch.Tensor) or matrix.numel() == 0:
        return matrix
    try:
        return F.normalize(matrix, dim=1)
    except Exception as exc:
        safe_print_warn(f"[warn] gagal menormalisasi embedding vault: {exc}")
        return matrix


# =========================
# Global init (lazy load vault/embeddings)
# =========================
_VAULT_CONTENT: List[str] | None = None
_VAULT_EMB: torch.Tensor | None = None
_VAULT_EMB_NORM: torch.Tensor | None = None


def _ensure_vault_ready() -> None:
    """Load vault content/embeddings on first use to avoid heavy import-time side effects."""
    global _VAULT_CONTENT, _VAULT_EMB, _VAULT_EMB_NORM
    if _VAULT_CONTENT is not None and _VAULT_EMB is not None:
        return
    safe_print_info("Loading vault content...")
    _VAULT_CONTENT = load_vault(DEFAULT_VAULT)
    _VAULT_EMB = build_vault_embeddings(_VAULT_CONTENT, DEFAULT_EMBED_MODEL, DEFAULT_VAULT)
    _VAULT_EMB_NORM = _normalize_embedding_matrix(_VAULT_EMB)

def _timestamp() -> str:
    return datetime.now(WIB).isoformat(timespec="seconds")


def _make_message(
    role: str,
    content: str,
    *,
    timestamp: str | None = None,
    ai_readable: bool = True,
) -> Dict[str, Any]:
    return {
        "role": (role or "unknown").strip() or "unknown",
        "content": content,
        "timestamp": timestamp or _timestamp(),
        "ai_readable": bool(ai_readable),
    }


def _normalize_message(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        role = (raw.get("role") or "unknown").strip() or "unknown"
        content = str(raw.get("content") or "")
        timestamp = str(raw.get("timestamp") or _timestamp())
        ai_readable = bool(raw.get("ai_readable", True))
    else:
        role = "unknown"
        content = str(raw)
        timestamp = _timestamp()
        ai_readable = True
    return _make_message(role, content, timestamp=timestamp, ai_readable=ai_readable)


def _default_thread_state() -> Dict[str, Any]:
    return {
        "messages": [],
        "ai_paused": False,
        "handoff_reason": "",
        "handoff_ts": "",
        "seen_message_ids": [],
    }


def _normalize_thread_state(raw: Any) -> Dict[str, Any]:
    state = _default_thread_state()
    if isinstance(raw, dict):
        state["ai_paused"] = bool(raw.get("ai_paused", state["ai_paused"]))
        state["handoff_reason"] = str(raw.get("handoff_reason") or "")
        state["handoff_ts"] = str(raw.get("handoff_ts") or "")
        messages = raw.get("messages", [])
        seen_ids = raw.get("seen_message_ids", [])
    elif isinstance(raw, list):
        messages = raw
        seen_ids = []
    else:
        messages = []
        seen_ids = []
    state["messages"] = [_normalize_message(msg) for msg in messages]
    state["seen_message_ids"] = [str(i) for i in seen_ids if i]
    return state


# Per-thread lock to prevent concurrent writes to the shelve store for the same WA ID.
_THREAD_LOCKS: Dict[str, Lock] = {}
_THREAD_LOCKS_GUARD = Lock()


def _get_thread_lock(wa_id: str) -> Lock:
    key = (wa_id or "").strip() or "__default__"
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _THREAD_LOCKS[key] = lock
    return lock


def _ensure_thread_store_dir() -> None:
    """Create threads_db_store directory if missing."""
    try:
        Path(THREADS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.exception("Failed to create threads_db_store directory")


def _is_duplicate_message(message_id: str | None, state: Dict[str, Any]) -> bool:
    """Return True if we've already processed this message id for the thread."""
    if not message_id:
        return False
    seen = state.get("seen_message_ids") or []
    if message_id in seen:
        return True
    seen.append(message_id)
    if len(seen) > DEFAULT_MAX_SEEN_IDS:
        del seen[:-DEFAULT_MAX_SEEN_IDS]
    state["seen_message_ids"] = seen
    return False


def _load_thread_state(wa_id: str) -> Dict[str, Any]:
    _ensure_thread_store_dir()
    with shelve.open(THREADS_DB_PATH) as db:
        raw = db.get(wa_id)
    return _normalize_thread_state(raw)


def _save_thread_state(wa_id: str, state: Dict[str, Any]) -> None:
    payload = {
        "messages": state.get("messages", []),
        "ai_paused": bool(state.get("ai_paused")),
        "handoff_reason": state.get("handoff_reason") or "",
        "handoff_ts": state.get("handoff_ts") or "",
        "seen_message_ids": state.get("seen_message_ids", []),
    }
    _ensure_thread_store_dir()
    with shelve.open(THREADS_DB_PATH) as db:
        db[wa_id] = payload


def pause_thread_for_manual_message(
    wa_id: str,
    message_body: str | None = None,
    *,
    timestamp: str | None = None,
    reason: str | None = None,
) -> None:
    """
    Pause AI via contact store when an outbound human message is detected.
    """
    if not wa_id:
        return
    _ = message_body
    _ = timestamp
    pause_reason = reason or "manual_outbound_message"
    try:
        contact = get_contact(wa_id)
    except Exception:
        contact = None
    if contact and not contact.get("allow_bot", True):
        logging.info("Skipping manual pause for %s because allow_bot is already false.", wa_id)
        return
    try:
        upsert_contact(
            wa_id,
            category="other",
            allow_bot=False,
            source="manual_outbound",
            routing_reason=pause_reason,
            routing_model="manual_outbound",
        )
    except Exception:
        logging.exception("Failed to pause via contact store for %s", wa_id)
        return
    logging.info("Paused AI for %s via contact store.", wa_id)


def _guard_message_length(cleaned_body: str, wa_id: str) -> str | None:
    """Return early message if the input fails length checks; otherwise None."""
    if len(cleaned_body) < DEFAULT_MIN_MSG_LENGTH:
        message = dedent(
            """
            Mohon maaf, pesan Anda perlu lebih dari 40 karakter ya. 
            Silakan tambahkan sedikit penjelasan atau kata kunci agar saya bisa membantu dengan lebih tepat. 🙂
            """
        ).strip()
        return _log_and_reply(wa_id, cleaned_body, message)
    if len(cleaned_body) > DEFAULT_MAX_MSG_LENGTH:
        message = dedent(
            """
            Mohon maaf, pesan Anda terlalu panjang. 
            Silakan kirim pesan yang lebih singkat agar saya dapat memprosesnya dengan baik. 🙂
            """
        ).strip()
        return _log_and_reply(wa_id, cleaned_body, message)
    return None


def _guard_prompt_injection(cleaned_body: str, wa_id: str) -> str | None:
    """Return early message if a prompt injection attempt is detected; otherwise None."""
    if not contains_prompt_injection_attempt(cleaned_body):
        return None
    logging.info("Mendeteksi upaya prompt injection dari %s: %s", wa_id, cleaned_body)
    guard_reply = prompt_injection_response()
    return _log_and_reply(wa_id, cleaned_body, guard_reply)


def _handle_ai_paused(
    state: Dict[str, Any],
    cleaned_body: str,
    wa_id: str,
    timestamp: str,
) -> str | None:
    """If conversation is paused for human assistance, log the message and return empty reply."""
    if not state["ai_paused"]:
        return None
    state["messages"].append(
        _make_message("user", cleaned_body, timestamp=timestamp, ai_readable=False)
    )
    _save_thread_state(wa_id, state)
    logging.info(
        "Skipping automated reply for %s because the conversation is paused for human assistance.",
        wa_id,
    )
    return ""


def _handle_handoff_detection(
    state: Dict[str, Any],
    cleaned_body: str,
    wa_id: str,
    timestamp: str,
    *,
    needs_handoff: bool,
    classifier_reason: str,
) -> str | None:
    """
    Run LLM-based handoff detection. If handoff is needed, update state and return handoff reply.
    """
    if state["ai_paused"]:
        return None

    if not needs_handoff:
        return None

    history = state["messages"]
    handoff_reason = (
        classifier_reason or "LLM classified message as human assistance request."
    )
    history.append(
        _make_message("user", cleaned_body, timestamp=timestamp, ai_readable=False)
    )
    state["ai_paused"] = True
    state["handoff_reason"] = handoff_reason
    state["handoff_ts"] = timestamp
    handoff_reply = _make_message(
        "assistant", HUMAN_HANDOFF_RESPONSE, ai_readable=True
    )
    history.append(handoff_reply)
    try:
        upsert_contact(
            wa_id,
            allow_bot=False,
            source="handoff_detect",
            routing_reason=handoff_reason,
            routing_model=DEFAULT_GEN_MODEL,
        )
    except Exception:
        logging.debug("Failed to persist handoff flag to contact store for %s", wa_id)
    _save_thread_state(wa_id, state)
    logging.info("Thread %s marked for human assistance: %s", wa_id, handoff_reason)
    return HUMAN_HANDOFF_RESPONSE


def _log_and_reply(
    wa_id: str,
    user_text: str,
    reply_text: str,
    *,
    user_ai_readable: bool = False,
    reply_ai_readable: bool = False,
) -> str:
    """
    Append the incoming user text and immediate reply to the thread state, then persist it.
    Used for early-return guardrails (message length, prompt injection, etc.).
    """
    state = _load_thread_state(wa_id)
    timestamp = _timestamp()
    state["messages"].append(
        _make_message(
            "user",
            user_text,
            timestamp=timestamp,
            ai_readable=user_ai_readable,
        )
    )
    state["messages"].append(
        _make_message(
            "system",
            reply_text,
            timestamp=timestamp,
            ai_readable=reply_ai_readable,
        )
    )
    _save_thread_state(wa_id, state)
    return reply_text


# =========================
# Public entrypoint for your webhook
# =========================
def generate_response(message_body: str, wa_id: str, message_id: str | None = None) -> str:
    """
    Drop-in replacement for your previous OpenAI Assistants-based generate_response.
    - message_body: text from WhatsApp user
    - wa_id: sender's WhatsApp ID (used to persist conversation history)
    Returns: assistant reply (string)
    """
    lock = _get_thread_lock(wa_id)
    with lock:
        cleaned_body = (message_body or "").strip()

        try:
            contact = get_contact(wa_id)
        except Exception:
            contact = None
        if contact and not contact.get("allow_bot", True):
            logging.info(
                "Skipping automated reply for %s because allow_bot is disabled (category=%s, source=%s).",
                wa_id,
                contact.get("category", ""),
                contact.get("source", ""),
            )
            return ""

        length_guard_reply = _guard_message_length(cleaned_body, wa_id)
        if length_guard_reply is not None:
            return length_guard_reply

        injection_guard_reply = _guard_prompt_injection(cleaned_body, wa_id)
        if injection_guard_reply is not None:
            return injection_guard_reply

        _ensure_vault_ready()
        if _VAULT_EMB is None or _VAULT_CONTENT is None:
            logging.error("Vault embeddings not available; skipping automated reply.")
            return "Maaf, sedang ada gangguan pada sistem kami. Coba lagi sebentar ya."

        state = _load_thread_state(wa_id)
        history = state["messages"]
        user_history_text = _collect_user_history_text(history)

        system_message = OPTIMAXX_SYSTEM_PROMPT if not history else None
        timestamp = _timestamp()

        if _is_duplicate_message(message_id, state):
            logging.info("Skipping duplicate/old message %s for %s", message_id, wa_id)
            _save_thread_state(wa_id, state)
            return ""

        paused_reply = _handle_ai_paused(state, cleaned_body, wa_id, timestamp)
        if paused_reply is not None:
            return paused_reply

        combined_checks = _run_combined_checks(
            cleaned_body,
            user_history_text,
            model=DEFAULT_GEN_MODEL,
        )
        
        handoff_reply = _handle_handoff_detection(
            state,
            cleaned_body,
            wa_id,
            timestamp,
            needs_handoff=bool(combined_checks.get("needs_human", False)),
            classifier_reason=combined_checks.get("handoff_reason", ""),
        )
        if handoff_reply is not None:
            return handoff_reply

        try:
            user_entry = _make_message(
                "user", cleaned_body, timestamp=timestamp, ai_readable=True
            )
            answer = chat_with_rag(
                user_input=message_body,
                system_message=system_message,
                vault_embeddings=_VAULT_EMB,
                vault_embeddings_norm=_VAULT_EMB_NORM,
                vault_content=_VAULT_CONTENT,
                ollama_model=DEFAULT_GEN_MODEL,
                embed_model=DEFAULT_EMBED_MODEL,
                conversation_history=history,
                user_entry=user_entry,
                precomputed_checks=combined_checks,
                top_k=DEFAULT_TOP_K,
                min_similarity=DEFAULT_MIN_SIMILARITY,
                num_ctx=DEFAULT_NUM_CTX,
                temperature=DEFAULT_GEN_TEMPERATURE,
                top_p=DEFAULT_GEN_TOP_P,
                repeat_penalty=DEFAULT_GEN_REPEAT_PENALTY,
            )
            if not answer:
                logging.info(
                    "Skipping reply for %s because no relevant context was found.", wa_id
                )
                return ""
            # Persist updated history
            state["messages"] = history
            state["handoff_reason"] = state.get("handoff_reason", "")
            state["handoff_ts"] = state.get("handoff_ts", "")
            _save_thread_state(wa_id, state)
            logging.info("Generated message for %s: %s", wa_id, answer)
            return answer
        except Exception as e:
            logging.exception("Failed to generate response: %s", e)
            return "Maaf, sedang ada gangguan pada sistem kami. Coba lagi sebentar ya."
