// import * as Moonshine from "https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-js@latest/dist/moonshine.min.js"
// uncomment to use latest build in this repo
import * as Moonshine from "@moonshine-ai/moonshine-js"

const state = document.getElementById("state")
const button = document.getElementById("button")
const history = document.getElementById("history")
const historyWrapper = document.getElementById("historyWrapper")

var pastText = ""

const partialUpdates = true // transcribers will use streaming mode, rather than VAD chunks

const callbacks = {
    onPermissionsRequested() {
        state.innerHTML = "Requesting permissions."
    },
    onError(e) {
        if (e == Moonshine.MoonshineError.PermissionDenied) {
            state.innerHTML = "Permission denied."
        } else {
            state.innerHTML = "An error occurred."
        }
    },
    onModelLoadStarted() {
        state.innerHTML = "Model loading."
    },
    onModelLoaded() {
        state.innerHTML = "Model loaded."
    },
    onTranscribeStarted() {
        state.innerHTML = "Started. Waiting for speech."
    },
    onSpeechStart() {
        console.log("Speech started.");
        state.innerHTML = "Speech started. Listening."
    },
    onSpeechEnd() {
        console.log("Speech ended.");
        state.innerHTML = "Speech ended."
    },
    onSpeechContinuing(audio) {
    },
    onTranscriptionCommitted(text) {
        console.log("Transcript committed:", text);
        if (text) {
            state.innerHTML = "Transcript committed."
            pastText = `${pastText} <br> ${text}`
            history.innerHTML = pastText
            historyWrapper.scrollTo({
                top: historyWrapper.scrollHeight,
                behavior: 'smooth'
            });
        }
    },
    onTranscriptionUpdated(text, audio) {
        console.log("Transcript updated:", text);
        if (text) {
            state.innerHTML = "Transcript updated."
            history.innerHTML = `${pastText} <br> ${text}`
            historyWrapper.scrollTo({
                top: historyWrapper.scrollHeight,
                behavior: 'smooth'
            });
        }
    },
    onTranscribeStopped() {
        state.innerHTML = "Stopped."
    },
}

Moonshine.Settings.BASE_ASSET_PATH.AUDIO_WORKLET = "audio-capture.worklet.js"

var microphoneTranscriber = new Moonshine.MicrophoneTranscriber(
    "model/base",
    callbacks,
    partialUpdates
);
// Start loading the models before the user clicks the button. This
// isn't strictly necessary since .start() will load the model if it's not
// already loaded, but it's a good idea to do it in advance so that the user
// doesn't have to wait for the model to load when they click the button.
microphoneTranscriber.load();

button.addEventListener("click", () => {
    if (microphoneTranscriber.isActive) {
        microphoneTranscriber.stop()
        button.innerText = "Start"
    }
    else {
        microphoneTranscriber.start()
        button.innerText = "Stop"
    }
})

var audioTranscriber = new Moonshine.MediaElementTranscriber(
    document.getElementById("audio"),
    "model/base",
    callbacks,
    partialUpdates 
);
audioTranscriber.load();

var videoCaptioner = new Moonshine.VideoCaptioner(
    document.getElementById("video"),
    "model/base",
    partialUpdates
)
videoCaptioner.load();
