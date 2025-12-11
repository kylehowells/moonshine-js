const frameSize = 512
const updateInterval = 16
const vadCommitSeconds = 10

/**
 * Global settings for MoonshineJS.
 */
export const Settings = {
    FRAME_SIZE: frameSize, // as specified by silero v5; changing this is not recommended
    TEN_VAD_FRAME_SIZE: 256,
    TEN_VAD_THRESHOLD: 0.5,
    VAD_PROBABILITY_WINDOW_SIZE: 32, // in frames
    VAD_LOOK_BEHIND_SAMPLE_COUNT: 8192,
    STT_MINIMUM_INTERVAL_MS: 200, // Don't run the STT model more than once every 200ms.
    SPEECH_MAX_DURATION_MS: 30000, // After this duration of continuous speech, the transcriber will commit the transcription regardless of whether the user is still speaking.
    BASE_ASSET_PATH: {
        MOONSHINE: "https://download.moonshine.ai/",
        ONNX_RUNTIME: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
        TEN_VAD_WASM: "https://download.moonshine.ai/scripts/",
        AUDIO_WORKLET: "https://download.moonshine.ai/scripts/audio-capture.worklet.js"
    },
    VERBOSE_LOGGING: false
}
