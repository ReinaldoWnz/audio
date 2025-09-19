import streamlit as st
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
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

    st.write("🔄 Carregando modelo de remoção de ruído...")
    # Modelo correto para enhancement (Speech Enhancement)
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/denoiser/mimic-voicebank",
        savedir="pretrained_model",
        run_opts={"device":"cpu"}  # CPU para Streamlit Cloud
    )

    st.write("🔄 Processando áudio...")
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
