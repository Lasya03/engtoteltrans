import streamlit as st
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# =============================
# 📦 Load dataset (optional)
# =============================
ds = load_dataset("prudhvirajdowluri/english_to_telugu_translation")

# =============================
# 🧠 Load model and tokenizer
# =============================
st.write("🔹 Loading model... Please wait ⏳")

model_name = "aryaumesh/english-to-telugu"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# =============================
# 💬 Translation function
# =============================
def translate(text):
    if not text.strip():
        return "⚠️ Please enter some text."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
# 🌐 Streamlit UI
# =============================
st.title("🌏 English → Telugu Translator")
st.write("Instant translation powered by Hugging Face Transformers 🚀")

user_input = st.text_area("Enter English text here 👇", height=100)

if st.button("Translate"):
    with st.spinner("Translating... ⏳"):
        translation = translate(user_input)
    st.success("✅ Translation:")
    st.write(f"**{translation}**")

# Optional: show examples
st.sidebar.header("Examples 🧩")
examples = [
    "Hello, how are you?",
    "What are you doing?",
    "Do you know what happened yesterday?",
    "How is Indra movie?",
]
example = st.sidebar.selectbox("Try one:", examples)
if st.sidebar.button("Translate Example"):
    st.sidebar.write("Telugu →", translate(example))
