import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import tempfile
import os

st.title("🎙️ Limpeza de Áudio com Noisereduce")

# Upload do arquivo de áudio
uploaded_file = st.file_uploader("Faça upload de um arquivo de áudio (WAV ou MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Salva arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    st.write("🔄 Carregando e processando áudio...")

    # Carrega áudio
    y, sr = librosa.load(input_path, sr=None)

    # Estima ruído (primeiros 0.5s)
    noise_sample = y[0:int(sr*0.5)]

    # Aplica redução de ruído
    reduced_noise = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr)

    # Salva áudio processado
    output_path = input_path.replace(".wav", "_clean.wav")
    sf.write(output_path, reduced_noise, sr)

    st.success("✅ Áudio processado com sucesso!")

    # Player para ouvir resultado
    st.audio(output_path, format="audio/wav")

    # Botão para download
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Baixar áudio limpo",
            data=f,
            file_name="audio_clean.wav",
            mime="audio/wav"
        )

    # Limpeza dos arquivos temporários
    os.remove(input_path)
