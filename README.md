# Random Forest `.pkl` + Streamlit

โปรเจกต์ตัวอย่างสำหรับสอนกระบวนการ Machine Learning แบบครบวงจร

- Data Preprocessing
- Train/Test Split
- Transform Data
- Random Forest Classification
- Model Evaluation
- Save Model เป็น `.pkl`
- Streamlit Web Application

## ไฟล์ภายในโครงการ

```text
random_forest_pkl_streamlit_project/
├── Random_Forest_Colab_PKL.ipynb
├── student_performance.csv
├── random_forest_model.pkl
├── model_metrics.json
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## วิธีใช้บน Google Colab

1. เปิดไฟล์ `Random_Forest_Colab_PKL.ipynb`
2. อัปโหลด `student_performance.csv`
3. รันเซลล์ตามลำดับ
4. ดาวน์โหลด `random_forest_model.pkl`
5. นำไฟล์ `.pkl` ไปวางในโฟลเดอร์เดียวกับ `app.py`

## วิธีรันเว็บ Streamlit

ติดตั้งไลบรารี

```bash
pip install -r requirements.txt
```

รันเว็บ

```bash
streamlit run app.py
```

## การ Deploy บน Streamlit Community Cloud

อัปโหลดไฟล์ต่อไปนี้ขึ้น GitHub

```text
app.py
random_forest_model.pkl
requirements.txt
runtime.txt
```

จากนั้นสร้างแอปและเลือก `app.py` เป็น Main file

## การเปลี่ยนเป็นข้อมูลจริง

แก้ส่วนต่อไปนี้ใน Notebook

```python
TARGET_COLUMN = "ชื่อคอลัมน์คำตอบ"

NUMERIC_FEATURES = [
    "คอลัมน์ตัวเลข1",
    "คอลัมน์ตัวเลข2",
]

CATEGORICAL_FEATURES = [
    "คอลัมน์หมวดหมู่1",
]
```

จากนั้นแก้ช่องรับข้อมูลใน `app.py` ให้ตรงกับคอลัมน์ของข้อมูลจริง

## ข้อควรระวัง

- แบ่ง Train/Test ก่อน Fit ตัวแปลงข้อมูล เพื่อป้องกัน Data Leakage
- บันทึก Preprocessor และ Model รวมกันใน Pipeline
- ใช้เวอร์ชัน scikit-learn ใกล้เคียงกันระหว่าง Colab และ Streamlit
- โหลดไฟล์ `.pkl` เฉพาะไฟล์ที่สร้างเองหรือมาจากแหล่งที่เชื่อถือได้
