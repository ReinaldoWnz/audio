import streamlit as st
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
from huggingface_hub import snapshot_download
import tempfile
import os

# Título do app
st.title("🎙️ Melhorar Áudio com SpeechBrain")

# Upload do arquivo de áudio
uploaded_file = st.file_uploader("Faça upload de um arquivo de áudio (WAV ou MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    # Baixa modelo do Hugging Face Hub
    st.write("🔄 Baixando modelo SpeechBrain (primeira vez pode demorar)...")
    model_dir = snapshot_download("speechbrain/mtl-mimic-voicebank")

    # Carrega modelo
    st.write("🔄 Processando áudio...")
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source=model_dir,
        savedir="pretrained_model",
        run_opts={"device": "cpu"}  # Streamlit Cloud normalmente só tem CPU
    )

    # Carrega áudio
    noisy, fs = torchaudio.load(input_path)

    # Aplica melhoria
    enhanced = enhance_model.enhance_batch(noisy, fs)

    # Salva resultado temporário
    output_path = input_path.replace(".wav", "_enhanced.wav")
    torchaudio.save(output_path, enhanced, fs)

    st.success("✅ Áudio processado com sucesso!")

    # Player para ouvir resultado
    st.audio(output_path, format="audio/wav")

    # Link para download
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Baixar áudio melhorado",
            data=f,
            file_name="audio_enhanced.wav",
            mime="audio/wav"
        )

    # Limpeza
    os.remove(input_path)
