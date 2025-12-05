"""
🎙️ WhisperLiveKit Streamlit Launcher
Simple interface to start WhisperLiveKit transcription server.
"""

import streamlit as st
import subprocess
import sys
import os
import signal
import time
import webbrowser
import threading
import queue

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(page_title="WhisperLiveKit", page_icon="🎙️", layout="centered")

# ---------------------------
# Session State Initialization
# ---------------------------
if "server_process" not in st. session_state:
    st.session_state.server_process = None
if "server_running" not in st. session_state:
    st.session_state.server_running = False
if "show_expert_mode" not in st.session_state:
    st.session_state.show_expert_mode = False
if "server_port" not in st. session_state:
    st.session_state.server_port = 8000
if "server_url" not in st.session_state:
    st.session_state.server_url = None
if "server_logs" not in st.session_state:
    st.session_state.server_logs = []
if "log_queue" not in st.session_state:
    st. session_state.log_queue = queue. Queue()
if "server_pid" not in st. session_state:
    st.session_state.server_pid = None

# ---------------------------
# Constants
# ---------------------------
LANGUAGES = {
    "Auto-detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Polish": "pl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Greek": "el",
    "Czech": "cs",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Norwegian": "no",
    "Ukrainian": "uk",
    "Hebrew": "he",
    "Indonesian": "id",
    "Malay": "ms",
}

MODEL_LIST = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

BACKEND_OPTIONS = [
    "Auto",
    "Faster-Whisper",
    "MLX-Whisper (Mac)",
    "OpenAI Whisper",
    "OpenAI API",
]

# ---------------------------
# UI Header
# ---------------------------
st.title("🎙️ WhisperLiveKit")
st.caption("Real-time speech transcription")

# ---------------------------
# Sidebar: Server Status
# ---------------------------
with st.sidebar:
    st.header("Status")
    if st.session_state.server_running and st.session_state.server_url:
        st.success("🟢 Server Running")
        st.markdown(
            f"**URL:** [{st.session_state.server_url}]({st.session_state.server_url})"
        )
        if st.button("🔗 Open URL", key="sidebar_open"):
            webbrowser.open(st.session_state. server_url)
    else:
        st.error("🔴 Server Stopped")

# ---------------------------
# Configuration
# ---------------------------
st.header("⚙️ Configuration")

# Model Selection
model_index = 5  # default to "large-v3-turbo"
model_size = st. selectbox(
    "Model",
    MODEL_LIST,
    index=model_index,
    help="Larger models are more accurate but slower.  'base' is recommended for most users.",
)

# Language Selection
language_keys = list(LANGUAGES.keys())
language_index = 4  # default to german
language_display = st.selectbox(
    "Language",
    language_keys,
    index=language_index,
    help="Select the language being spoken, or use auto-detect.",
)
language = LANGUAGES[language_display]

# Features
col1, col2 = st.columns(2)
with col1:
    enable_translate = st.checkbox(
        "Translate speech in real-time", value=False, help="Enable translation"
    )
with col2:
    enable_diarization = st. checkbox(
        "Speaker Detection",
        value=True,
        help="Identify different speakers (requires additional setup)",
    )

# Port
port = st.number_input(
    "Port",
    value=8000,
    min_value=1024,
    max_value=65535,
    help="Port number for the server",
)

# Advanced Options
with st.expander("🔧 Advanced Options"):
    backend_policy = st.selectbox(
        "Streaming Mode",
        ["SimulStreaming (Faster)", "LocalAgreement (More Stable)"],
        index=0,
        help="SimulStreaming is faster; LocalAgreement is more stable",
    )
    backend = st.selectbox(
        "Whisper Backend",
        BACKEND_OPTIONS,
        index=0,
        help="Auto selects the best available backend",
    )

# ---------------------------
# Translation Menu
# ---------------------------
st.subheader("🌐 Translation")
translation_mode = st. selectbox(
    "Translation Mode",
    ["Off", "Translate to English", "Translate to Specific Language"],
    index=0,
    help="Choose how transcription should be translated",
)

translate_to_english = False
target_language_override = ""

if translation_mode == "Translate to English":
    translate_to_english = True
elif translation_mode == "Translate to Specific Language":
    target_lang_index = 1  # Default to English
    target_language_override_display = st.selectbox(
        "Target Language",
        language_keys,
        index=target_lang_index,
        help="Choose the language to translate into",
    )
    target_language_override = LANGUAGES[target_language_override_display]

# ---------------------------
# Expert Mode
# ---------------------------
st.divider()
if st.button(
    "⚠️ Expert Mode - Only if you know what you're doing!", use_container_width=True
):
    st.session_state.show_expert_mode = not st.session_state.show_expert_mode
    st. rerun()

# Default Expert Values
expert_defaults = {
    "host": "127.0.0.1",
    "log_level": "INFO",
    "ssl_certfile": "",
    "ssl_keyfile": "",
    "forwarded_allow_ips": "",
    "target_language": "",
    "model_cache_dir": "",
    "model_dir": "",
    "lora_path": "",
    "warmup_file": "jfk. flac",
    "buffer_trimming": "segment",
    "buffer_trimming_sec": 15.0,
    "confidence_validation": False,
    "no_vac": False,
    "no_vad": False,
    "pcm_input": False,
    "no_transcription": False,
    "punctuation_split": False,
    "diarization_backend": "sortformer",
    "segmentation_model": "pyannote/segmentation-3.0",
    "embedding_model": "pyannote/embedding",
    "min_chunk_size": 0.1,
    "vac_chunk_size": 0.04,
    "frame_threshold": 25,
    "beams": 1,
    "decoder_type": "auto",
    "audio_max_len": 30.0,
    "audio_min_len": 0.0,
    "max_context_tokens": 175,
    "disable_fast_encoder": False,
    "never_fire": False,
    "init_prompt": "",
    "static_init_prompt": "",
    "model_path": "",
    "cif_ckpt_path": "",
    "custom_alignment_heads": "",
    "nllb_backend": "transformers",
    "nllb_size": "600M",
}

if st.session_state.show_expert_mode:
    st.warning(
        "⚠️ Expert Mode Active - Changing settings incorrectly may break the server!"
    )

    with st.expander("🖥️ Server Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            expert_defaults["host"] = st.text_input(
                "Host", value=expert_defaults["host"]
            )
        with col2:
            st.info(f"Port: {port} (set above)")
        with col3:
            log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            expert_defaults["log_level"] = st. selectbox(
                "Log Level", log_level_options, index=1
            )
        col1, col2 = st.columns(2)
        with col1:
            expert_defaults["ssl_certfile"] = st.text_input(
                "SSL Certificate File", value=""
            )
        with col2:
            expert_defaults["ssl_keyfile"] = st.text_input("SSL Key File", value="")
        expert_defaults["forwarded_allow_ips"] = st.text_input(
            "Forwarded Allow IPs", value=""
        )

    with st.expander("🤖 Model Configuration", expanded=False):
        expert_defaults["target_language"] = st.text_input("Target Language", value="")
        col1, col2 = st.columns(2)
        with col1:
            expert_defaults["model_cache_dir"] = st.text_input(
                "Model Cache Directory", value=""
            )
        with col2:
            expert_defaults["model_dir"] = st.text_input("Model Directory", value="")
        expert_defaults["lora_path"] = st.text_input("LoRA Path", value="")
        expert_defaults["warmup_file"] = st.text_input("Warmup File", value="")

    with st.expander("⚙️ Backend Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            buffer_options = ["segment", "sentence"]
            expert_defaults["buffer_trimming"] = st. selectbox(
                "Buffer Trimming Strategy", buffer_options, index=0
            )
        with col2:
            expert_defaults["buffer_trimming_sec"] = st. slider(
                "Buffer Trimming Threshold (sec)",
                1.0,
                30.0,
                expert_defaults["buffer_trimming_sec"],
                0.5,
            )

    with st.expander("✨ Feature Flags", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            expert_defaults["confidence_validation"] = st.checkbox(
                "Confidence Validation", value=False
            )
        with col2:
            expert_defaults["no_vac"] = st.checkbox("Disable VAC", value=False)
        with col3:
            expert_defaults["no_vad"] = st.checkbox("Disable VAD", value=False)
        with col4:
            expert_defaults["pcm_input"] = st.checkbox("PCM Input", value=False)
        col1, col2 = st.columns(2)
        with col1:
            expert_defaults["no_transcription"] = st.checkbox(
                "Disable Transcription", value=False
            )
        with col2:
            expert_defaults["punctuation_split"] = st.checkbox(
                "Punctuation Split", value=False
            )

    if enable_diarization:
        with st.expander("🎭 Diarization Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                diar_options = ["sortformer", "diart"]
                expert_defaults["diarization_backend"] = st.selectbox(
                    "Diarization Backend", diar_options, index=0
                )
            with col2:
                expert_defaults["segmentation_model"] = st.text_input(
                    "Segmentation Model", value=expert_defaults["segmentation_model"]
                )
            with col3:
                expert_defaults["embedding_model"] = st.text_input(
                    "Embedding Model", value=expert_defaults["embedding_model"]
                )

    with st.expander("🎵 Audio & Streaming", expanded=False):
        col1, col2 = st. columns(2)
        with col1:
            expert_defaults["min_chunk_size"] = st.slider(
                "Min Chunk Size (sec)",
                0.01,
                1.0,
                expert_defaults["min_chunk_size"],
                0.01,
            )
        with col2:
            expert_defaults["vac_chunk_size"] = st.slider(
                "VAC Chunk Size (sec)",
                0.01,
                0.2,
                expert_defaults["vac_chunk_size"],
                0.01,
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            expert_defaults["frame_threshold"] = st. number_input(
                "Frame Threshold", value=expert_defaults["frame_threshold"], min_value=1
            )
            expert_defaults["beams"] = st. number_input(
                "Beams", value=expert_defaults["beams"], min_value=1
            )
            decoder_options = ["auto", "beam", "greedy"]
            expert_defaults["decoder_type"] = st.selectbox(
                "Decoder Type", decoder_options, index=0
            )
        with col2:
            expert_defaults["audio_max_len"] = st.slider(
                "Audio Max Length (sec)",
                1.0,
                60.0,
                expert_defaults["audio_max_len"],
                1.0,
            )
            expert_defaults["audio_min_len"] = st.slider(
                "Audio Min Length (sec)",
                0.0,
                5.0,
                expert_defaults["audio_min_len"],
                0.1,
            )
            expert_defaults["max_context_tokens"] = st.number_input(
                "Max Context Tokens",
                value=expert_defaults["max_context_tokens"],
                min_value=0,
            )
        with col3:
            expert_defaults["disable_fast_encoder"] = st.checkbox(
                "Disable Fast Encoder", value=False
            )
            expert_defaults["never_fire"] = st.checkbox("Never Fire (CIF)", value=False)

        col1, col2 = st.columns(2)
        with col1:
            nllb_backend_options = ["transformers", "ctranslate2"]
            expert_defaults["nllb_backend"] = st.selectbox(
                "NLLB Backend", nllb_backend_options, index=0
            )
        with col2:
            nllb_size_options = ["600M", "1. 3B"]
            expert_defaults["nllb_size"] = st. selectbox(
                "NLLB Size", nllb_size_options, index=0
            )


# ---------------------------
# Command Builder
# ---------------------------
def build_command():
    cmd = [sys.executable, "-m", "whisperlivekit.basic_server"]
    cmd.extend(["--host", expert_defaults["host"]])
    cmd. extend(["--port", str(port)])
    cmd.extend(["--model", model_size])
    cmd. extend(["--lan", language])
    cmd.extend(["--log-level", expert_defaults["log_level"]])

    policy = (
        "localagreement" if "LocalAgreement" in backend_policy else "simulstreaming"
    )
    cmd.extend(["--backend-policy", policy])

    backend_map = {
        "Auto": "auto",
        "Faster-Whisper": "faster-whisper",
        "MLX-Whisper (Mac)": "mlx-whisper",
        "OpenAI Whisper": "whisper",
        "OpenAI API": "openai-api",
    }
    cmd.extend(["--backend", backend_map. get(backend, "auto")])

    optional_strings = [
        ("--ssl-certfile", expert_defaults["ssl_certfile"]),
        ("--ssl-keyfile", expert_defaults["ssl_keyfile"]),
        ("--forwarded-allow-ips", expert_defaults["forwarded_allow_ips"]),
        ("--target-language", expert_defaults["target_language"]),
        ("--model_cache_dir", expert_defaults["model_cache_dir"]),
        ("--model_dir", expert_defaults["model_dir"]),
        ("--lora-path", expert_defaults["lora_path"]),
        ("--warmup-file", expert_defaults["warmup_file"]),
        ("--init-prompt", expert_defaults["init_prompt"]),
        ("--static-init-prompt", expert_defaults["static_init_prompt"]),
        ("--model-path", expert_defaults["model_path"]),
        ("--cif-ckpt-path", expert_defaults["cif_ckpt_path"]),
        ("--custom-alignment-heads", expert_defaults["custom_alignment_heads"]),
    ]
    for flag, val in optional_strings:
        if val:
            cmd. extend([flag, val])

    if translate_to_english:
        cmd.append("--direct-english-translation")
    elif target_language_override:
        cmd.extend(["--target-language", target_language_override])

    if enable_diarization:
        cmd.append("--diarization")
        cmd.extend(["--diarization-backend", expert_defaults["diarization_backend"]])
        cmd.extend(["--segmentation-model", expert_defaults["segmentation_model"]])
        cmd. extend(["--embedding-model", expert_defaults["embedding_model"]])

    flags = [
        "confidence_validation",
        "no_vac",
        "no_vad",
        "pcm_input",
        "no_transcription",
        "punctuation_split",
        "disable_fast_encoder",
        "never_fire",
    ]
    for f in flags:
        if expert_defaults[f]:
            cmd.append(f"--{f. replace('_', '-')}")

    cmd.extend(["--frame-threshold", str(expert_defaults["frame_threshold"])])
    cmd.extend(["--beams", str(expert_defaults["beams"])])
    if expert_defaults["decoder_type"] != "auto":
        cmd.extend(["--decoder", expert_defaults["decoder_type"]])
    cmd.extend(["--audio-max-len", str(expert_defaults["audio_max_len"])])
    cmd.extend(["--audio-min-len", str(expert_defaults["audio_min_len"])])
    if expert_defaults["max_context_tokens"] > 0:
        cmd.extend(["--max-context-tokens", str(expert_defaults["max_context_tokens"])])
    cmd.extend(["--nllb-backend", expert_defaults["nllb_backend"]])
    cmd. extend(["--nllb-size", expert_defaults["nllb_size"]])
    cmd.extend(["--min-chunk-size", str(expert_defaults["min_chunk_size"])])
    cmd.extend(["--vac-chunk-size", str(expert_defaults["vac_chunk_size"])])
    cmd.extend(["--buffer_trimming", expert_defaults["buffer_trimming"]])
    cmd. extend(["--buffer_trimming_sec", str(expert_defaults["buffer_trimming_sec"])])

    return cmd


# ---------------------------
# Log Reader Thread
# ---------------------------
def read_output(process, log_queue):
    """Read process output in a separate thread."""
    try:
        for line in iter(process.stdout. readline, ""):
            if line:
                log_queue.put(line. strip())
            if process.poll() is not None:
                break
    except Exception:
        pass


# ---------------------------
# Process Killing
# ---------------------------
def kill_process_tree(pid):
    """Kill a process and all its children.  Works on Windows and Unix."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal. SIGTERM)
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass


def kill_port(port_num):
    """Kill any process using the specified port."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10
            )
            for line in result. stdout.split("\n"):
                if f":{port_num}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit():
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                timeout=10,
                            )
        except Exception:
            pass
    else:
        try:
            subprocess. run(
                ["fuser", "-k", f"{port_num}/tcp"], capture_output=True, timeout=10
            )
        except Exception:
            try:
                result = subprocess.run(
                    ["lsof", "-t", "-i", f":{port_num}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for pid in result.stdout. strip().split("\n"):
                    if pid.isdigit():
                        os.kill(int(pid), signal. SIGKILL)
            except Exception:
                pass


# ---------------------------
# Server Start / Stop
# ---------------------------
def start_server():
    """Start the server and return the process and log queue."""
    cmd = build_command()

    st.session_state.server_logs = []

    log_queue = queue. Queue()
    st.session_state. log_queue = log_queue

    if os.name != "nt":
        process = subprocess.Popen(
            cmd,
            stdout=subprocess. PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
    else:
        process = subprocess. Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    st.session_state. server_process = process
    st.session_state.server_pid = process.pid
    st.session_state.server_port = port

    log_thread = threading.Thread(
        target=read_output, args=(process, log_queue), daemon=True
    )
    log_thread. start()

    return process, log_queue


def stop_server():
    """Stop the server - forcefully if needed."""
    stopped = False

    if st.session_state. server_process is not None:
        try:
            pid = st.session_state.server_process.pid
            kill_process_tree(pid)
            try:
                st.session_state. server_process.wait(timeout=3)
                stopped = True
            except subprocess.TimeoutExpired:
                pass

            if st.session_state. server_process.poll() is None:
                try:
                    st.session_state. server_process.kill()
                    st.session_state. server_process.wait(timeout=2)
                    stopped = True
                except Exception:
                    pass
        except Exception:
            pass

    if st.session_state. server_port:
        kill_port(st.session_state.server_port)
        stopped = True

    if st.session_state. server_pid:
        kill_process_tree(st.session_state.server_pid)
        stopped = True

    st.session_state.server_process = None
    st.session_state.server_pid = None
    st. session_state.server_running = False
    st.session_state.server_url = None
    st.session_state.server_logs = []

    return stopped


def get_new_logs():
    """Get any new logs from the queue."""
    new_logs = []
    try:
        while not st.session_state.log_queue. empty():
            line = st.session_state.log_queue.get_nowait()
            new_logs.append(line)
            st.session_state. server_logs.append(line)
    except Exception:
        pass
    return new_logs


# ---------------------------
# Launch Controls
# ---------------------------
st.divider()

if not st.session_state.server_running:
    # Show command preview before starting
    st.subheader("📋 Command to Execute")
    cmd = build_command()
    st.code(" ".join(cmd), language="bash")
    st.caption("💡 You can copy this command to run manually in a terminal if needed.")

    st.divider()

    if st.button(
        "🚀 Start Transcription Server", type="primary", use_container_width=True
    ):
        process, log_queue = start_server()

        st.info("🔄 Starting server...  This may take a while for large models.")
        log_placeholder = st.empty()
        status_placeholder = st. empty()

        server_ready = False
        start_time = time. time()

        while not server_ready:
            elapsed = int(time.time() - start_time)
            status_placeholder.text(f"⏳ Loading... ({elapsed}s elapsed)")

            try:
                while not log_queue.empty():
                    line = log_queue.get_nowait()
                    st.session_state. server_logs.append(line)

                    recent_logs = st. session_state.server_logs[-15:]
                    log_placeholder.code("\n".join(recent_logs), language="text")

                    if (
                        "Uvicorn running on" in line
                        or "Application startup complete" in line
                    ):
                        server_ready = True
                        if "http://" in line:
                            start_idx = line.find("http://")
                            end_idx = line. find(" ", start_idx)
                            if end_idx == -1:
                                end_idx = len(line)
                            st.session_state.server_url = line[
                                start_idx:end_idx
                            ].strip()
                        else:
                            st.session_state. server_url = f"http://localhost:{port}"
                        break
            except Exception:
                pass

            if server_ready:
                break

            if process.poll() is not None:
                st.error(
                    "❌ Server process terminated unexpectedly. Check the logs above."
                )
                st.session_state.server_running = False
                st.stop()

            time.sleep(0.3)

        st.session_state.server_running = True
        if not st.session_state.server_url:
            st.session_state.server_url = f"http://localhost:{port}"

        status_placeholder.empty()
        st. success("✅ Server started successfully!")
        time.sleep(1)
        st.rerun()

else:
    url = st.session_state.server_url
    st.success("✅ Server is running!")
    st. markdown("### 🌐 Open the transcription interface:")
    st.markdown(f"**[{url}]({url})**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Open in Browser", use_container_width=True):
            webbrowser.open(url)
    with col2:
        if st. button("⏹️ Stop Server", type="secondary", use_container_width=True):
            with st.spinner("Stopping server..."):
                stopped = stop_server()
                time.sleep(1)
            if stopped:
                st.success("Server stopped.")
            st.rerun()

    with st.expander("📜 Server Logs", expanded=False):
        get_new_logs()

        if st.session_state.server_logs:
            recent_logs = st. session_state.server_logs[-50:]
            st. code("\n".join(recent_logs), language="text")
        else:
            st.info("No logs available yet.")

        if st.button("🔄 Refresh Logs"):
            get_new_logs()
            st.rerun()

    # Show running command
    with st.expander("📋 Running Command", expanded=False):
        cmd = build_command()
        st.code(" ".join(cmd), language="bash")
        if st.session_state.server_pid:
            st.caption(f"Process ID (PID): {st.session_state. server_pid}")

# ---------------------------
# Footer Help
# ---------------------------
st.divider()
with st.expander("ℹ️ Help"):
    st.markdown(
        """
### How to use
1. Select model size (larger = more accurate but slower)
2.  Choose the language being spoken
3. Enable translation if desired
4. Review the command that will be executed
5. Click **Start Transcription Server**
6. Wait for the server to start (you'll see logs)
7. Open the provided URL in your browser

### Model Recommendations
- **tiny/base**: Fast, good for real-time
- **small/medium**: More accurate
- **large-v3 / large-v3-turbo**: Very accurate, slower

### Stopping the Server
If the Stop button doesn't work, you can:
- **Windows**: Open Task Manager, find `python.exe` and end the task
- **Windows**: Run `taskkill /F /IM python.exe` in Command Prompt
- **Linux/Mac**: Run `pkill -f whisperlivekit` in terminal

### Notes
- Expert mode allows advanced configuration - only change if you understand the options. 
- Default translation is OFF. 
- Server logs are displayed during startup and can be viewed while running.
- Large models may take several minutes to load. 
    """
    )