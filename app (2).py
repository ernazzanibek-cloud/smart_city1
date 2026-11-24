import os
import re
import sqlite3
import uuid
from datetime import datetime

import folium
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from streamlit_folium import st_folium

# ---------------- Баптаулар ----------------
DB_PATH = "reports.db"
UPLOAD_DIR = "uploads"
DEFAULT_CENTER = (51.1694, 71.4491)  # Астана. Өз қалаңа ауыстыр: (ендік, бойлық)
ADMIN_PASSWORD = "admin123"           # Әкімші панелінің паролі
STATUSES = ["Жаңа", "Жұмыста", "Орындалды"]
CATEGORIES = ["Жол", "Жарық", "Қоқыс", "Көгалдандыру"]
PRIORITY_COLORS = {"Жоғары": "red", "Орташа": "orange", "Төмен": "green"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Деректер қоры ----------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT, address TEXT, photo TEXT,
                lat REAL, lon REAL,
                category TEXT, priority TEXT, score INTEGER,
                status TEXT, created TEXT)"""
        )


def load_reports():
    with get_conn() as c:
        return pd.read_sql_query("SELECT * FROM reports ORDER BY id DESC", c)


def add_report(row):
    with get_conn() as c:
        c.execute(
            """INSERT INTO reports
               (text, address, photo, lat, lon, category, priority, score, status, created)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            row,
        )


def update_status(report_id, status):
    with get_conn() as c:
        c.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))


# ---------------- NLP: мәтін түрін анықтау ----------------
TRAIN = {
    "Жол": [
        "жолда үлкен шұңқыр бар", "асфальт бұзылған", "көше жолы жөнделмеген",
        "яма на дороге", "разбитый асфальт", "дорога в плохом состоянии",
        "светофор не работает", "светофор істемейді", "жол белгісі құлаған",
        "тротуар сынған", "люк ашық тұр", "открытый люк на дороге",
    ],
    "Жарық": [
        "көшеде шамдар жанбайды", "түнде жарық өшіп қалады", "фонарь не горит",
        "уличное освещение не работает", "бағандағы шам сынған",
        "ауладағы жарық жоқ", "в темном дворе нет света", "лампа перегорела",
        "электр бағаны жарық бермейді", "нет освещения на улице",
    ],
    "Қоқыс": [
        "қоқыс жәшігі толып кетті", "қоқыс шығарылмайды", "мусор не вывозят",
        "переполненный контейнер", "стихийная свалка", "көшеде қоқыс үйіліп жатыр",
        "контейнер жанында қоқыс", "қоқыс иісі шығады", "тастанды қоқыс орны",
        "свалка во дворе", "куча мусора",
    ],
    "Көгалдандыру": [
        "ағаш сынған", "бұтақтар жолға құлады", "шөп шабылмаған",
        "газон не скошен", "дерево упало", "сухие деревья нужно убрать",
        "ағаш отырғызу керек", "посадить деревья", "газон бұзылған",
        "в сквере трава вытоптана", "көгалдандыру қажет", "клумбы заброшены",
    ],
}


@st.cache_resource
def train_text_model():
    texts, labels = [], []
    for cat, samples in TRAIN.items():
        for s in samples:
            texts.append(s.lower())
            labels.append(cat)
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(texts, labels)
    return model


def classify_text(text):
    model = train_text_model()
    probs = model.predict_proba([text.lower()])[0]
    idx = probs.argmax()
    return model.classes_[idx], float(probs[idx])


# ---------------- Computer Vision: фото түрін анықтау (міндетті емес) ----------------
CLIP_PROMPTS = {
    "Жол": "a photo of a damaged road with potholes",
    "Жарық": "a photo of a broken street light at night",
    "Қоқыс": "a photo of garbage and trash piles",
    "Көгалдандыру": "a photo of fallen trees or an overgrown lawn",
}


@st.cache_resource
def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name), CLIPProcessor.from_pretrained(name)


def classify_image(img):
    """CLIP zero-shot. transformers/torch орнатылмаса None қайтарады."""
    try:
        import torch

        model, processor = load_clip()
        cats = list(CLIP_PROMPTS.keys())
        inputs = processor(
            text=list(CLIP_PROMPTS.values()), images=img,
            return_tensors="pt", padding=True,
        )
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
        idx = int(probs.argmax())
        return cats[idx], float(probs[idx])
    except Exception:
        return None


# ---------------- Маңыздылықты бағалау ----------------
HIGH_WORDS = ["қауіпті", "опасно", "апат", "авария", "ашық люк", "открыт", "жарақат",
              "травм", "құлады", "упал", "өрт", "пожар", "газ", "бала", "дет"]
MID_WORDS = ["ұзақ уақыт", "давно", "әлі", "несколько недель", "апта", "неделю", "көп"]
BASE_SCORE = {"Жол": 2, "Жарық": 2, "Қоқыс": 1, "Көгалдандыру": 1}


def assess_priority(text, category, has_photo, lat, lon):
    t = text.lower()
    score = BASE_SCORE.get(category, 1)
    score += 2 * sum(w in t for w in HIGH_WORDS)
    score += 1 * sum(w in t for w in MID_WORDS)
    if has_photo:
        score += 1
    # Жақын жерде осындай өтініштер көп болса, маңыздылығы артады
    df = load_reports()
    if not df.empty:
        near = df[
            (df.category == category)
            & ((df.lat - lat).abs() < 0.001)
            & ((df.lon - lon).abs() < 0.0015)
            & (df.status != "Орындалды")
        ]
        score += min(len(near), 3)
    level = "Жоғары" if score >= 5 else "Орташа" if score >= 3 else "Төмен"
    return level, score


# ---------------- Геокодтау (OpenStreetMap / Nominatim) ----------------
@st.cache_data(show_spinner=False)
def geocode(address):
    try:
        geo = Nominatim(user_agent="smart_city_student_project")
        loc = geo.geocode(address, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return None


# ---------------- Беттер ----------------
def page_submit():
    st.header("📝 Өтініш беру")
    with st.form("report_form", clear_on_submit=True):
        text = st.text_area("Мәселені сипаттаңыз", placeholder="Мысалы: Абай көшесінде шұңқыр бар, қауіпті")
        address = st.text_input("Мекенжай", placeholder="Қала, көше, үй нөмірі")
        photo = st.file_uploader("Фото (міндетті емес)", type=["jpg", "jpeg", "png"])
        use_clip = st.checkbox("Фото арқылы да түрін анықтау (CLIP, transformers қажет)")
        submitted = st.form_submit_button("Жіберу")

    if not submitted:
        return
    if not text.strip() and photo is None:
        st.error("Мәтін жазыңыз немесе фото жүктеңіз.")
        return

    # 1) Түрін анықтау
    category, conf = (None, 0.0)
    if text.strip():
        category, conf = classify_text(text)

    photo_path = None
    if photo is not None:
        img = Image.open(photo).convert("RGB")
        photo_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
        img.save(photo_path)
        if use_clip and (category is None or conf < 0.5):
            res = classify_image(img)
            if res:
                category, conf = res
            else:
                st.warning("CLIP жұмыс істемеді (transformers/torch орнатылмаған). Мәтін негізінде анықталды.")
    if category is None:
        category, conf = "Жол", 0.0

    # 2) Картаға орналастыру
    coords = geocode(address) if address.strip() else None
    if coords is None:
        st.warning("Мекенжай табылмады, қала орталығы қолданылды.")
        coords = DEFAULT_CENTER
    lat, lon = coords

    # 3) Маңыздылық
    priority, score = assess_priority(text, category, photo_path is not None, lat, lon)

    add_report((
        text, address, photo_path, lat, lon, category, priority, score,
        "Жаңа", datetime.now().strftime("%Y-%m-%d %H:%M"),
    ))
    st.success("Өтініш қабылданды ✅")
    st.write(f"**Түрі:** {category} (сенімділік: {conf:.0%})")
    st.write(f"**Маңыздылығы:** {priority} (балл: {score})")
    st.write(f"**Координаттар:** {lat:.5f}, {lon:.5f}")


def page_map():
    st.header("🗺️ Мәселелер картасы")
    df = load_reports()
    if df.empty:
        st.info("Әзірге өтініш жоқ.")
        return

    c1, c2 = st.columns(2)
    cats = c1.multiselect("Түрі", CATEGORIES, default=CATEGORIES)
    stats = c2.multiselect("Мәртебе", STATUSES, default=STATUSES)
    df = df[df.category.isin(cats) & df.status.isin(stats)]

    m = folium.Map(location=DEFAULT_CENTER, zoom_start=12)
    for _, r in df.iterrows():
        popup = (
            f"<b>#{r.id} {r.category}</b><br>{r.text}<br>"
            f"Маңыздылығы: {r.priority}<br>Мәртебе: {r.status}<br>{r.created}"
        )
        folium.Marker(
            [r.lat, r.lon],
            popup=folium.Popup(popup, max_width=250),
            tooltip=f"{r.category} · {r.status}",
            icon=folium.Icon(color=PRIORITY_COLORS.get(r.priority, "blue")),
        ).add_to(m)
    st_folium(m, height=500, use_container_width=True, returned_objects=[])

    st.caption("🔴 Жоғары  🟠 Орташа  🟢 Төмен")
    st.dataframe(
        df[["id", "category", "priority", "status", "address", "created"]],
        use_container_width=True, hide_index=True,
    )


def page_admin():
    st.header("🛠️ Әкімші панелі")
    pwd = st.text_input("Пароль", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Кіру үшін пароль енгізіңіз.")
        return

    df = load_reports()
    if df.empty:
        st.info("Өтініш жоқ.")
        return

    # Статистика
    c1, c2, c3 = st.columns(3)
    c1.metric("Барлығы", len(df))
    c2.metric("Орындалуда", int((df.status == "Жұмыста").sum()))
    c3.metric("Орындалды", int((df.status == "Орындалды").sum()))
    st.bar_chart(df.category.value_counts())

    st.subheader("Мәртебені өзгерту")
    rid = st.selectbox("Өтініш №", df.id.tolist())
    row = df[df.id == rid].iloc[0]
    st.write(f"**{row.category}** · {row.priority} · {row.address}")
    st.write(row.text)
    if row.photo and os.path.exists(row.photo):
        st.image(row.photo, width=300)
    new_status = st.selectbox("Жаңа мәртебе", STATUSES, index=STATUSES.index(row.status))
    if st.button("Сақтау"):
        update_status(int(rid), new_status)
        st.success("Жаңартылды")
        st.rerun()


# ---------------- ИИ көмекші ----------------
ASSISTANT_MODEL = "claude-sonnet-5-5"  # қаласаң "claude-haiku-5-5" (арзанырақ, жылдамырақ)

SYSTEM_PROMPT = """Сен Smart City платформасының ИИ көмекшісісің.
Міндетіңіз:
- Тұрғынға қала мәселесін (жол, жарық, қоқыс, көгалдандыру) нақты сипаттауға көмектесу:
  не болды, қай жерде (мекенжай), қашаннан бері, қауіпті ме.
- Мәселенің қай санатқа жататынын айту және "📝 Өтініш беру" бетінде өтініш жіберуге бағыттау.
- Өтініш нөмірі берілсе, жүйе деректеріндегі мәртебені айту.
- Платформаның қалай жұмыс істейтінін түсіндіру (карта, маңыздылық, мәртебелер).
Ережелер: тұрғын қай тілде жазса, сол тілде (қазақша немесе орысша) қысқа және түсінікті жауап бер.
Жүйе деректерінде жоқ мәліметті ойдан шығарма, білмесең, солай де.
Қала мәселесіне қатысы жоқ сұрақтарға қысқаша жауап беріп, негізгі тақырыпқа қайтар."""

GREETING = (
    "Сәлем! Мен Smart City ИИ көмекшісімін. Қала мәселесін сипаттаңыз "
    "(жол, жарық, қоқыс, көгалдандыру) немесе өтініш нөмірін жазыңыз, мәртебесін айтамын."
)


GEMINI_MODEL = "gemini-2.5-flash"  # тегін деңгей; қолжетімді модельдерді aistudio.google.com ішінен қара


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def get_api_key():
    return get_secret("GEMINI_API_KEY") or get_secret("ANTHROPIC_API_KEY")


def ask_gemini(key, history, system):
    import requests

    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in history[-9:]  # соңғы хабарламалар (user-ден басталады)
    ]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"system_instruction": {"parts": [{"text": system}]}, "contents": contents},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def lookup_reports(user_text):
    """Хабарламадағы сандар бойынша өтініш мәртебесін табады."""
    df = load_reports()
    lines = []
    if df.empty:
        return lines
    for rid in set(re.findall(r"\d{1,6}", user_text)):
        row = df[df.id == int(rid)]
        if not row.empty:
            r = row.iloc[0]
            lines.append(
                f"Өтініш №{r.id}: түрі {r.category}, маңыздылығы {r.priority}, "
                f"мәртебесі {r.status}, күні {r.created}."
            )
    return lines


def build_context(user_text):
    df = load_reports()
    lines = [f"Жүйедегі өтініштер саны: {len(df)}."]
    if not df.empty:
        lines.append("Түрлері бойынша: " + ", ".join(f"{k}: {v}" for k, v in df.category.value_counts().items()))
        lines.append("Мәртебелер бойынша: " + ", ".join(f"{k}: {v}" for k, v in df.status.value_counts().items()))
    return "\n".join(lines + lookup_reports(user_text))


def fallback_answer(user_text):
    """API кілті жоқ болса, жүйенің өз классификаторымен жауап береді."""
    cat, conf = classify_text(user_text)
    parts = [
        f"Хабарламаңыз «{cat}» санатына жақын (сенімділік: {conf:.0%}). "
        "«📝 Өтініш беру» бетінде мәтін, мекенжай және фото енгізіп өтініш жіберіңіз."
    ]
    parts += lookup_reports(user_text)
    return "\n\n".join(parts)


def ask_assistant(history, user_text):
    gemini_key = get_secret("GEMINI_API_KEY")
    claude_key = get_secret("ANTHROPIC_API_KEY")
    if not gemini_key and not claude_key:
        return fallback_answer(user_text)
    system = SYSTEM_PROMPT + "\n\nЖүйе деректері:\n" + build_context(user_text)
    try:
        if gemini_key:  # тегін Google Gemini
            return ask_gemini(gemini_key, history, system)
        import anthropic

        client = anthropic.Anthropic(api_key=claude_key)
        resp = client.messages.create(
            model=ASSISTANT_MODEL,
            max_tokens=600,
            system=system,
            messages=history[-9:],  # соңғы хабарламалар (user-ден басталады)
        )
        return resp.content[0].text
    except Exception as e:
        return fallback_answer(user_text) + f"\n\n(ИИ қосылмады: {type(e).__name__})"


def page_assistant():
    st.header("🤖 ИИ көмекші")
    if not get_api_key():
        st.info("ИИ кілті (ANTHROPIC_API_KEY) жоқ, қарапайым режим жұмыс істеп тұр.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.chat_message("assistant"):
        st.write(GREETING)
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Мәселеңізді жазыңыз...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Ойланып жатыр..."):
                answer = ask_assistant(st.session_state.chat, prompt)
            st.write(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})


# ---------------- Іске қосу ----------------
st.set_page_config(page_title="Smart City", page_icon="🏙️", layout="wide")
init_db()
st.title("🏙️ Smart City: қала мәселелерін тіркеу жүйесі")

page = st.sidebar.radio("Мәзір", ["📝 Өтініш беру", "🗺️ Карта", "🤖 ИИ көмекші", "🛠️ Әкімші"])
if page.startswith("📝"):
    page_submit()
elif page.startswith("🗺️"):
    page_map()
elif page.startswith("🤖"):
    page_assistant()
else:
    page_admin()
