# ================================================================
#  app.py — ระบบทำนายแนวโน้มการสอบผ่านด้วย Random Forest
# ================================================================
#  โครงสร้างโปรแกรม: แยกเป็นฟังก์ชันตามหน้าที่ ชัดเจนอ่านง่าย
#    - load_bundle()        โหลดโมเดลจากไฟล์ .pkl (มี cache)
#    - inject_theme()       ตกแต่งหน้าตาด้วย CSS
#    - page_single()        หน้าทำนายรายบุคคล
#    - page_batch()         หน้าทำนายจากไฟล์ CSV
#    - page_pipeline()      หน้าอธิบายขั้นตอนของโมเดล
#    - main()               จุดเริ่มโปรแกรม + เมนูนำทาง
#  การนำทางใช้เมนูในแถบข้าง (radio) แทนแท็บ เลือกหน้าได้จากที่เดียว
# ================================================================

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

# ---------------- ค่าคงที่ของระบบ ----------------
MODEL_FILE = Path(__file__).with_name("random_forest_model.pkl")

PAGES = ("ทำนายรายบุคคล", "ทำนายจาก CSV", "ขั้นตอนของโมเดล")

# แถวตัวอย่างสำหรับไฟล์ CSV และคำอธิบายคอลัมน์ (ใช้ร่วมกันหลายหน้า)
SAMPLE_ROW = {
    "study_hours": 4.0,
    "attendance_percent": 85.0,
    "assignment_score": 75.0,
    "previous_gpa": 2.75,
    "internet_access": "Yes",
    "tutoring": "No",
}

COLUMN_GUIDE = [
    ("study_hours",        "ตัวเลข",   "ชั่วโมงอ่านหนังสือต่อวัน"),
    ("attendance_percent", "ตัวเลข",   "เปอร์เซ็นต์การเข้าเรียน"),
    ("assignment_score",   "ตัวเลข",   "คะแนนงานหรือการบ้าน"),
    ("previous_gpa",       "ตัวเลข",   "เกรดเฉลี่ยเดิม 0–4"),
    ("internet_access",    "หมวดหมู่", "Yes หรือ No"),
    ("tutoring",           "หมวดหมู่", "Yes หรือ No"),
]


# ================================================================
#  ส่วนโหลดโมเดล
# ================================================================
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_bundle() -> dict:
    """โหลดไฟล์โมเดล (.pkl) ที่บันทึกไว้ พร้อม pipeline และ metadata

    ใช้ @st.cache_resource เพื่อให้โหลดจากดิสก์เพียงครั้งเดียว
    ทุกการรีรันหลังจากนั้นจะใช้ตัวที่อยู่ในหน่วยความจำทันที
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {MODEL_FILE.name} "
            "กรุณาวางไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับ app.py"
        )
    # โหลดเฉพาะไฟล์ .pkl ที่สร้างจากแหล่งที่เชื่อถือได้เท่านั้น
    with open(MODEL_FILE, "rb") as fh:
        return pickle.load(fh)


# ================================================================
#  ส่วนตกแต่งหน้าตา (CSS)
# ================================================================
def inject_theme() -> None:
    """ธีม 'กระดาษรายงาน' — พื้นเรียบ การ์ดขอบบาง หัวข้อมีขีดสีคั่น"""
    st.markdown(
        """
        <style>
            .block-container { max-width: 1080px; padding-top: 1.6rem; }

            /* หัวเรื่องหลัก: ขีดสีด้านซ้ายแทนกล่องไล่เฉดแบบทั่วไป */
            .title-block {
                border-left: 5px solid #4f46e5;
                padding: .2rem 0 .2rem 1.1rem;
                margin-bottom: 1.4rem;
            }
            .title-block h1 { font-size: 1.75rem; margin: 0; }
            .title-block p  { margin: .25rem 0 0 0; opacity: .72; }

            /* การ์ดผลลัพธ์ */
            .result-shell {
                border: 1px solid rgba(120,120,140,.25);
                border-radius: 14px;
                padding: 1.1rem 1.25rem;
            }

            div[data-testid="stMetric"] {
                border: 1px solid rgba(120,120,140,.2);
                border-radius: 12px;
                padding: .7rem .9rem;
            }

            .dim-note { font-size: .85rem; opacity: .65; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    """หัวเรื่องประจำหน้า รูปแบบเดียวกันทุกหน้าเพื่อความสม่ำเสมอ"""
    st.markdown(
        f"""
        <div class="title-block">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
#  หน้า 1 — ทำนายรายบุคคล
# ================================================================
def page_single(pipeline, metadata) -> None:
    """ฟอร์มกรอกข้อมูลนักเรียน 1 คน แล้วแสดงผลทำนายด้านขวา"""
    page_header(
        "🎓 ทำนายรายบุคคล",
        "กรอกข้อมูลนักเรียน ระบบจะประเมินแนวโน้มการสอบผ่านทันที",
    )

    # เลย์เอาต์ 2 ฝั่ง: ซ้ายคือฟอร์ม / ขวาคือผลลัพธ์ (ต่างจากแบบฟอร์มบน-ผลล่าง)
    form_col, result_col = st.columns([6, 5], gap="large")

    with form_col:
        with st.form("single_form"):
            st.markdown("**ข้อมูลด้านการเรียน**")
            study_hours = st.number_input(
                "ชั่วโมงอ่านหนังสือต่อวัน",
                min_value=0.0, max_value=12.0, value=4.0, step=0.5,
            )
            attendance_percent = st.number_input(
                "เปอร์เซ็นต์การเข้าเรียน",
                min_value=0.0, max_value=100.0, value=85.0, step=1.0,
            )
            assignment_score = st.number_input(
                "คะแนนงานหรือการบ้าน",
                min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            )
            previous_gpa = st.number_input(
                "เกรดเฉลี่ยเดิม",
                min_value=0.0, max_value=4.0, value=2.75, step=0.05,
            )

            st.markdown("**ปัจจัยสนับสนุนการเรียน**")
            internet_access = st.selectbox(
                "มีอินเทอร์เน็ตสำหรับการเรียนหรือไม่",
                options=["Yes", "No"],
                format_func=lambda v: "มี" if v == "Yes" else "ไม่มี",
            )
            tutoring = st.selectbox(
                "เข้าร่วมการสอนเสริมหรือไม่",
                options=["Yes", "No"],
                format_func=lambda v: "เข้าร่วม" if v == "Yes" else "ไม่เข้าร่วม",
            )

            submitted = st.form_submit_button(
                "ทำนายผล", type="primary", use_container_width=True,
            )

    with result_col:
        if not submitted:
            # สถานะเริ่มต้น: บอกผู้ใช้ว่าต้องทำอะไรต่อ
            st.info("กรอกข้อมูลด้านซ้ายแล้วกด **ทำนายผล** ผลลัพธ์จะแสดงตรงนี้")
            return

        # รวมค่าจากฟอร์มเป็น DataFrame 1 แถว ตามคอลัมน์ที่โมเดลรู้จัก
        record = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percent": attendance_percent,
            "assignment_score": assignment_score,
            "previous_gpa": previous_gpa,
            "internet_access": internet_access,
            "tutoring": tutoring,
        }])

        predicted_class = int(pipeline.predict(record)[0])
        proba_fail, proba_pass = (float(p) for p in pipeline.predict_proba(record)[0])
        label = metadata["class_names"][predicted_class]

        st.markdown('<div class="result-shell">', unsafe_allow_html=True)
        if predicted_class == 1:
            st.success(f"ผลการทำนาย: **{label}**")
        else:
            st.warning(f"ผลการทำนาย: **{label}**")

        m1, m2 = st.columns(2)
        m1.metric("ความน่าจะเป็นที่จะผ่าน", f"{proba_pass:.1%}")
        m2.metric("ความน่าจะเป็นที่จะไม่ผ่าน", f"{proba_fail:.1%}")
        st.progress(proba_pass)

        st.markdown(
            '<p class="dim-note">ข้อมูลจะผ่านขั้นตอนเติมค่าที่หาย ปรับมาตรฐาน '
            'และแปลงข้อมูลหมวดหมู่โดยอัตโนมัติก่อนเข้าสู่ Random Forest</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
#  หน้า 2 — ทำนายจากไฟล์ CSV
# ================================================================
def page_batch(pipeline, metadata) -> None:
    """อัปโหลด CSV หลายรายการ ทำนายทั้งชุด พร้อมสรุปและดาวน์โหลดผล"""
    page_header(
        "📄 ทำนายจาก CSV",
        "เตรียมไฟล์ตามแบบ อัปโหลด แล้วรับผลการทำนายทั้งชุดพร้อมดาวน์โหลด",
    )

    # นำเสนอเป็นลำดับขั้น 1-2-3 ให้ผู้ใช้ทำตามได้ทันที
    st.markdown("##### ขั้นที่ 1 · เตรียมไฟล์ตามแบบ")
    template = pd.DataFrame([SAMPLE_ROW])
    st.download_button(
        label="ดาวน์โหลดไฟล์ CSV ตัวอย่าง",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    st.markdown("##### ขั้นที่ 2 · อัปโหลดไฟล์")
    uploaded = st.file_uploader(
        "อัปโหลดไฟล์ CSV",
        type=["csv"],
        help="ชื่อคอลัมน์ต้องตรงกับไฟล์ตัวอย่าง",
    )
    if uploaded is None:
        return

    try:
        rows = pd.read_csv(uploaded)

        # ตรวจคอลัมน์ให้ครบก่อนส่งเข้าโมเดล
        needed = metadata["feature_columns"]
        absent = [c for c in needed if c not in rows.columns]
        if absent:
            st.error("ไฟล์ขาดคอลัมน์ต่อไปนี้: " + ", ".join(absent))
            return

        predictions = pipeline.predict(rows[needed].copy())
        pass_scores = pipeline.predict_proba(rows[needed].copy())[:, 1]

        output = rows.copy()
        output["prediction"] = predictions
        output["prediction_label"] = [
            metadata["class_names"][int(v)] for v in predictions
        ]
        output["pass_probability"] = pass_scores.round(4)

        st.markdown("##### ขั้นที่ 3 · ผลการทำนาย")

        # สรุปภาพรวมเป็นตัวเลขก่อน แล้วค่อยแสดงตารางละเอียด
        total = len(output)
        passed = int((predictions == 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนทั้งหมด", f"{total:,}")
        c2.metric("คาดว่าผ่าน", f"{passed:,}")
        c3.metric("คาดว่าไม่ผ่าน", f"{total - passed:,}")

        st.dataframe(output, use_container_width=True, hide_index=True)

        st.download_button(
            label="ดาวน์โหลดผลการทำนาย",
            data=output.to_csv(index=False).encode("utf-8-sig"),
            file_name="random_forest_predictions.csv",
            mime="text/csv",
            type="primary",
        )
    except Exception as error:
        st.error(f"ไม่สามารถประมวลผลไฟล์ได้: {error}")


# ================================================================
#  หน้า 3 — ขั้นตอนของโมเดล
# ================================================================
def page_pipeline() -> None:
    """อธิบายกระบวนการภายในของ pipeline แบบกางอ่านทีละขั้น"""
    page_header(
        "⚙️ ขั้นตอนของโมเดล",
        "เส้นทางของข้อมูลตั้งแต่รับเข้า จนได้ผลการทำนาย",
    )

    # ใช้ expander แยกทีละขั้น อ่านเจาะเฉพาะส่วนที่สนใจได้
    with st.expander("ขั้นที่ 1 · เตรียมข้อมูลตัวเลข (Numeric Preprocessing)", expanded=True):
        st.markdown(
            "- เติมค่าที่หายด้วย **ค่ามัธยฐาน** ของแต่ละคอลัมน์\n"
            "- ปรับสเกลด้วย **StandardScaler** ให้ทุกตัวแปรอยู่ในช่วงเทียบกันได้"
        )
    with st.expander("ขั้นที่ 2 · เตรียมข้อมูลหมวดหมู่ (Categorical Preprocessing)", expanded=True):
        st.markdown(
            "- เติมค่าที่หายด้วย **ค่าที่พบบ่อยที่สุด**\n"
            "- แปลงเป็นตัวเลขด้วย **One-Hot Encoding**"
        )
    with st.expander("ขั้นที่ 3 · ทำนายผล (Prediction)", expanded=True):
        st.markdown(
            "- ส่งข้อมูลที่แปลงแล้วเข้าสู่ **Random Forest**\n"
            "- แสดงคลาสที่ทำนายพร้อม **ค่าความน่าจะเป็น** ของแต่ละคลาส"
        )

    st.markdown("##### โครงสร้างคอลัมน์ที่โมเดลรับ")
    st.dataframe(
        pd.DataFrame(COLUMN_GUIDE, columns=["ชื่อคอลัมน์", "ชนิดข้อมูล", "คำอธิบาย"]),
        use_container_width=True,
        hide_index=True,
    )


# ================================================================
#  จุดเริ่มโปรแกรม
# ================================================================
def main() -> None:
    st.set_page_config(
        page_title="Random Forest Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # โหลดโมเดลก่อนทำอย่างอื่น ถ้าไม่สำเร็จหยุดพร้อมข้อความบอกวิธีแก้
    try:
        bundle = load_bundle()
        pipeline, metadata = bundle["pipeline"], bundle["metadata"]
    except Exception as error:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {error}")
        st.stop()

    # ---------- แถบข้าง: เมนูนำทาง + ข้อมูลโมเดล ----------
    with st.sidebar:
        st.title("🎓 RF Predictor")
        chosen_page = st.radio("เมนู", PAGES, label_visibility="collapsed")

        st.divider()
        st.subheader("ข้อมูลโมเดล")
        st.write("อัลกอริทึม: **Random Forest**")
        st.write("ประเภทงาน: **Classification**")
        scores = metadata.get("metrics", {})
        st.metric("Test Accuracy", f"{scores.get('accuracy', 0):.2%}")
        st.metric("Test ROC-AUC", f"{scores.get('roc_auc', 0):.3f}")

        st.divider()
        st.caption(
            "โมเดลและข้อมูลนี้จัดทำเพื่อเป็นตัวอย่างการเรียนรู้ "
            "ไม่ควรใช้ตัดสินนักเรียนจริงโดยไม่มีการตรวจสอบเพิ่มเติม"
        )

    # ---------- แสดงหน้าตามเมนูที่เลือก ----------
    if chosen_page == PAGES[0]:
        page_single(pipeline, metadata)
    elif chosen_page == PAGES[1]:
        page_batch(pipeline, metadata)
    else:
        page_pipeline()


if __name__ == "__main__":
    main()