import { Settings } from "./constants";
import MoonshineModel from "./model";
import MoonshineError from "./error";
import Log from "./log";
import { AudioCapture } from "./audioCapture";
import { VAD } from "./vad";

/**
 * Callbacks are invoked at different phases of the lifecycle as audio is transcribed. You can control the behavior of the application
 * in response to model loading, starting of transcription, stopping of transcription, and updates to the transcription of the audio stream.
 *
 * @property onPermissionsRequested() - called when permissions to a user resource (e.g., microphone) have been requested (but not necessarily granted yet)
 *
 * @property onError(error: {@link MoonshineError}) - called when an error occurs.
 *
 * @property onModelLoadStarted() - called when the {@link MoonshineModel} and VAD begins to load (or download, if hosted elsewhere)
 *
 * @property onModelLoaded() - called when the {@link MoonshineModel} and VAD are loaded. This means the Transcriber is now ready to use.
 *
 * @property onTranscribeStarted() - called once when transcription starts
 *
 * @property onTranscribeStopped() - called once when transcription stops
 *
 * @property onTranscriptionUpdated(text: string) - when `partialUpdates === false` (i.e., streaming mode), this callback is invoked rapidly
 * with the speculative transcription of the current audio. It will not be called more than once every {@link Settings.STT_MINIMUM_INTERVAL_MS}.
 *
 * @property onTranscriptionCommitted(text: string, buffer?: AudioBuffer) - called every time a transcript is "committed"; when `partialUpdates === false` (streaming mode),
 * the transcript is committed between brief pauses in speech. When `partialUpdates === true`, the transcript is committed after speech events, or during brief pauses in long speech events.
 *
 * @property onSpeechStart() - called when the VAD model detects the start of speech
 *
 * @property onSpeechEnd() - called when the VAD model detects the end of speech
 *
 * @property onSpeechContinuing() - called when the VAD model detects continuing speech
 *
 * @interface
 */
interface TranscriberCallbacks {
    onPermissionsRequested: () => any;

    onError: (error) => any;

    onModelLoadStarted: () => any;

    onModelLoaded: () => any;

    onTranscribeStarted: () => any;

    onTranscribeStopped: () => any;

    onTranscriptionUpdated: (text: string, audio: Float32Array) => any;

    onTranscriptionCommitted: (text: string, audio: Float32Array) => any;

    onSpeechStart: () => any;

    onSpeechContinuing: (audio: Float32Array) => any;

    onSpeechEnd: (audio: Float32Array) => any;
}

const defaultTranscriberCallbacks: TranscriberCallbacks = {
    onPermissionsRequested: function () {
        Log.log("Transcriber.onPermissionsRequested()");
    },
    onError: function (error) {
        Log.error("Transcriber.onError(" + error + ")");
    },
    onModelLoadStarted: function () {
        Log.log("Transcriber.onModelLoadStarted()");
    },
    onModelLoaded: function () {
        Log.log("Transcriber.onModelLoaded()");
    },
    onTranscribeStarted: function () {
        Log.log("Transcriber.onTranscribeStarted()");
    },
    onTranscribeStopped: function () {
        Log.log("Transcriber.onTranscribeStopped()");
    },
    onTranscriptionUpdated: function (text: string, audio: Float32Array) {
        Log.log("Transcriber.onTranscriptionUpdated(" + text + ")");
    },
    onTranscriptionCommitted: function (text: string, audio: Float32Array) {
        Log.log("Transcriber.onTranscriptionCommitted(" + text + ")");
    },
    onSpeechStart: function () {
        Log.log("Transcriber.onSpeechStart()");
    },
    onSpeechContinuing: function (audio: Float32Array) {
        Log.log("Transcriber.onSpeechContinuing()");
    },
    onSpeechEnd: function (audio: Float32Array) {
        Log.log("Transcriber.onSpeechEnd()");
    },
};


/**
 * Implements real-time transcription of an audio stream sourced from a WebAudio-compliant MediaStream object.
 *
 * Read more about working with MediaStreams: {@link https://developer.mozilla.org/en-US/docs/Web/API/MediaStream}
 */
class Transcriber {
    private static models: Map<string, MoonshineModel> = new Map();
    private sttModel: MoonshineModel;
    private vad: VAD;
    private audioCapture: AudioCapture;
    private mediaStream: MediaStream;
    private isSttRunning: boolean = false;
    private lastSttFinishedTimeMs: number = 0;  // In milliseconds.
    callbacks: TranscriberCallbacks;

    private partialUpdates: boolean;
    private currentVoiceAudioBuffer: Float32Array;

    protected audioContext: AudioContext;
    public isActive: boolean = false;

    /**
     * Creates a transcriber for transcribing a MediaStream from any source. After creating the {@link Transcriber}, you must invoke
     * {@link Transcriber.attachStream} to provide a MediaStream that you want to transcribe.
     *
     * @param modelURL The URL that the underlying {@link MoonshineModel} weights should be loaded from,
     * relative to {@link Settings.BASE_ASSET_PATH.MOONSHINE}.
     *
     * @param callbacks A set of {@link TranscriberCallbacks} used to trigger behavior at different steps of the
     * transcription lifecycle. For transcription-only use cases, you should define the {@link TranscriberCallbacks} yourself;
     * when using the transcriber for voice control, you should create a {@link VoiceController} and pass it in.
     *
     * @param partialUpdates A boolean specifying whether to give partial transcriptions updates during speech.
     * When set to `true`, the transcriber will only process speech at the end of each chunk of voice activity; when set to `false`, the transcriber will
     * operate in streaming mode, generating continuous transcriptions on a rapid interval.
     *
     * @example
     * This basic example demonstrates the use of the transcriber with custom callbacks:
     *
     * ``` ts
     * import Transcriber from "@moonshine-ai/moonshine-js";
     *
     * var transcriber = new Transcriber(
     *      "model/tiny",
     *      {
     *          onModelLoadStarted() {
     *              console.log("onModelLoadStarted()");
     *          },
     *          onTranscribeStarted() {
     *              console.log("onTranscribeStarted()");
     *          },
     *          onTranscribeStopped() {
     *              console.log("onTranscribeStopped()");
     *          },
     *          onTranscriptionUpdated(text: string | undefined) {
     *              console.log(
     *                  "onTranscriptionUpdated(" + text + ")"
     *              );
     *          },
     *          onTranscriptionCommitted(text: string | undefined) {
     *              console.log(
     *                  "onTranscriptionCommitted(" + text + ")"
     *              );
     *          },
     *      },
     *      false // use streaming mode
     * );
     *
     * // Get a MediaStream from somewhere (user mic, active tab, an <audio> element, WebRTC source, etc.)
     * ...
     *
     * transcriber.attachStream(stream);
     * transcriber.start();
     * ```
     */
    public constructor(
        modelURL: string,
        callbacks: Partial<TranscriberCallbacks> = {},
        partialUpdates: boolean = true,
        precision: string = "quantized"
    ) {
        this.callbacks = { ...defaultTranscriberCallbacks, ...callbacks };
        // we want to avoid re-downloading the same model weights if we can avoid it
        // so we only create a new model of the requested type if it hasn't been already
        if (!Transcriber.models.has(modelURL))
            Transcriber.models.set(modelURL, new MoonshineModel(modelURL, precision));
        this.sttModel = Transcriber.models.get(modelURL);
        this.partialUpdates = partialUpdates;
        this.audioContext = new AudioContext();
    }

    /**
     * Preloads the models and initializes the buffer required for transcription.
     */
    public async load(): Promise<void> {
        this.callbacks.onModelLoadStarted();
        try {
            await this.sttModel.loadModel();
        } catch (err) {
            this.callbacks.onError(MoonshineError.PlatformUnsupported);
            throw err;
        }

        this.vad = new VAD({
            onVoiceStart: (audio: Float32Array) => {
                this.onVoiceStart(audio);
            },
            onVoiceEnd: (audio: Float32Array) => {
                this.onVoiceEnd(audio);
            },
            onVoiceContinuing: (audio: Float32Array) => {
                this.onVoiceContinuing(audio);
            },
            probabilityWindowSize: Settings.VAD_PROBABILITY_WINDOW_SIZE,
            lookBehindSampleCount: Settings.VAD_LOOK_BEHIND_SAMPLE_COUNT,
            threshold: Settings.TEN_VAD_THRESHOLD,
            frameSize: Settings.TEN_VAD_FRAME_SIZE,
            voiceMaxSampleCount: (Settings.SPEECH_MAX_DURATION_MS * 16000) / 1000, // Convert milliseconds to samples at 16000 Hz
        });
        await this.vad.load();
        this.callbacks.onModelLoaded();
    }

    /**
     * Attaches a MediaStream to this {@link Transcriber} for transcription. A MediaStream must be attached before
     * starting transcription.
     *
     * @param stream A MediaStream to transcribe
     */
    public attachStream(stream: MediaStream) {
        if (stream) {
            if (this.vad) {
                Log.log(
                    "Transcriber.attachStream(): VAD set to receive source node from stream."
                );
                this.audioCapture = new AudioCapture({
                    stream: stream,
                    audioContext: this.audioContext as AudioContext,
                    workletURL: Settings.BASE_ASSET_PATH.AUDIO_WORKLET,
                    frameSize: Settings.TEN_VAD_FRAME_SIZE,
                    onAudioCapture: (audio: Float32Array) => {
                        this.onAudioCapture(audio);
                    },
                    onStart: () => {
                        console.log("onStart");
                    },
                    onStop: () => {
                        console.log("onStop");
                    },
                });
                this.audioCapture.init();
            } else {
                // save stream to attach later, after loading
                this.mediaStream = stream;
            }
        }
    }

    /**
     * Detaches the MediaStream used for transcription.
     * TODO
     */
    public detachStream() {
        // TODO
    }

    /**
     * Returns the most recent AudioBuffer that was input to the underlying model for text generation. This is useful in cases where
     * we want to double-check the audio being input to the model while debugging.
     *
     * @returns An AudioBuffer
     */
    public getAudioBuffer(buffer: Float32Array): AudioBuffer {
        const numChannels = 1;
        const audioBuffer = this.audioContext.createBuffer(
            numChannels,
            buffer.length,
            16000
        );
        audioBuffer.getChannelData(0).set(buffer);
        return audioBuffer;
    }

    /**
     * Starts transcription.
     *
     * Transcription will stop when {@link stop} is called.
     *
     * Note that the {@link Transcriber} must have a MediaStream attached via {@link Transcriber.attachStream} before
     * starting transcription.
     */
    public async start() {
        if (!this.isActive) {
            this.isActive = true;

            // load model if not loaded
            if (
                (!this.sttModel.isLoaded() && !this.sttModel.isLoading()) ||
                this.vad === undefined
            ) {
                await this.load();
            }

            this.callbacks.onTranscribeStarted();
            this.vad.start();
            this.audioCapture.start();
            this.audioContext.resume();
            setTimeout(() => {
                if (this.audioContext.state === "suspended") {
                    Log.warn(
                        "AudioContext is suspended, this usually happens on Chrome when you start trying to access an audio source (like a microphone or video) before the user has interacted with the page. Chrome blocks access until there has been a user gesture, so you'll need to rework your code to call start() after an interaction."
                    );
                }
            }, 1000);
        }
    }

    /**
     * Stops transcription.
     */
    public stop() {
        this.isActive = false;
        this.callbacks.onTranscribeStopped();
        this.vad.stop();
        this.audioCapture.stop();
    }

    private onAudioCapture(audio_float32: Float32Array) {
        this.vad.processAudio(audio_float32);
    }

    private onVoiceStart(audio: Float32Array) {
        this.lastSttFinishedTimeMs = Date.now();
        this.currentVoiceAudioBuffer = audio;
        this.callbacks.onSpeechStart();
    }

    private onVoiceEnd(audio: Float32Array) {
        // Make a copy of this audio buffer so that the transcription callback captures
        // the current version, since the original audio buffer is passed by reference.
        const localAudioBuffer = Float32Array.from(audio);
        this.currentVoiceAudioBuffer = new Float32Array(0);
        this.callbacks.onSpeechEnd(localAudioBuffer);
        this.isSttRunning = true;
        this.sttModel?.generate(localAudioBuffer).then((text) => {
            this.callbacks.onTranscriptionCommitted(text, localAudioBuffer);
            this.lastSttFinishedTimeMs = Date.now();
            this.isSttRunning = false;
        });
    }

    private onVoiceContinuing(audio: Float32Array) {
        this.currentVoiceAudioBuffer = Float32Array.from(audio);
        // Make a copy of this audio buffer so that the transcription callback captures
        // the current version, since the original audio buffer is passed by reference.
        const localAudioBuffer = Float32Array.from(audio);
        this.callbacks.onSpeechContinuing(localAudioBuffer);
        if (this.isSttRunning || !this.partialUpdates) {
            return;
        }
        const currentTimeMs = Date.now();
        const timeSinceLastSttFinishedMs = currentTimeMs - this.lastSttFinishedTimeMs;
        // Put in a fallback in case the STT model errors out and doesn't hit the callback
        // that normally clears this flag.
        if (timeSinceLastSttFinishedMs > 10000) {
            this.isSttRunning = false;
        }
        // Don't run the STT model more often than it would be useful, to avoid spamming the client
        // with too many updates on faster devices.
        if (timeSinceLastSttFinishedMs < Settings.STT_MINIMUM_INTERVAL_MS) {
            return;
        }
        this.isSttRunning = true;
        this.sttModel?.generate(this.currentVoiceAudioBuffer).then((text) => {
            this.callbacks.onTranscriptionUpdated(text, localAudioBuffer);
            this.isSttRunning = false;
            this.lastSttFinishedTimeMs = Date.now();
        });
    }
}

export { Transcriber, TranscriberCallbacks };
