import * as ort from "onnxruntime-web";
import llamaTokenizer from "llama-tokenizer-js";
import { Settings } from "./constants";
import Log from "./log";
import { WordTiming, buildWordTimings } from "./wordAlignment";

function argMax(array) {
    return [].map
        .call(array, (x, i) => [x, i])
        .reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * Implements speech-to-text inferences with Moonshine models.
 */
export default class MoonshineModel {
    private modelURL: string;
    private precision: string;
    private model: any;

    private shape: any;
    private decoderStartTokenID: number = 1;
    private eosTokenID: number = 2;

    private lastLatency: number | undefined = undefined;
    private isModelLoading: boolean = false;
    private loadPromise: Promise<void>;

    /**
     * Create (but do not load) a new MoonshineModel for inference.
     *
     * @param modelURL A string (relative to {@link Settings.BASE_ASSET_PATH}) where the `.onnx` model weights are located.
     *
     * @remarks Creating a MoonshineModel has the side effect of setting the path to the `onnxruntime-web` `.wasm` to the {@link Settings.BASE_ASSET_PATH}
     */
    public constructor(inputModelURL: string, precision: string = "quantized") {
        // Switch to per-language naming convention for English models.
        let modelURL = inputModelURL;
        if (modelURL === "model/tiny") {
            modelURL = "model/tiny-en";
        } else if (modelURL === "model/base") {
            modelURL = "model/base-en";
        }
        this.modelURL = Settings.BASE_ASSET_PATH.MOONSHINE + modelURL;
        this.precision = precision;
        ort.env.wasm.wasmPaths = Settings.BASE_ASSET_PATH.ONNX_RUNTIME;
        this.model = {
            encoder: undefined,
            decoder: undefined,
        };
        if (this.modelURL.includes("tiny")) {
            this.shape = {
                numLayers: 6,
                numKVHeads: 8,
                headDim: 36,
            };
        } else if (this.modelURL.includes("base")) {
            this.shape = {
                numLayers: 8,
                numKVHeads: 8,
                headDim: 52,
            };
        }
        Log.log(`New MoonshineModel with modelURL = ${modelURL}`);
    }

    private static getSessionOption() {
        let sessionOption;

        // check for webgpu support
        // if (!!navigator.gpu) {
        //     sessionOption = {
        //         executionProviders: ["webgpu"],
        //         preferredOutputLocation: "gpu-buffer",
        //     };
        // }
        // otherwise check for webgl support
        // NOTE onnxruntime-web does not support the necessary ops for moonshine on webgl
        // else if (
        //     (function () {
        //         const canvas = document.createElement("canvas");
        //         return !!(
        //             canvas.getContext("webgl") || canvas.getContext("webgl2")
        //         );
        //     })()
        // ) {
        //     sessionOption = {
        //         executionProviders: ["webgl"]
        //     };
        // }
        // otherwise use cpu
        //else {
        sessionOption = {
            executionProviders: ["wasm", "cpu"],
        };
        //}
        return sessionOption;
    }

    /**
     * Tests the inference latency of the current environment.
     *
     * @remarks Warning: since this uses noise to benchmark the model, the model will have lower performance if you to use it
     * for transcription immediately after benchmarking.
     *
     * @param sampleSize (Optional) The number of samples to use for computing the benchmark
     *
     * @returns The average inference latency (in ms) over the number of samples taken.
     */
    public async benchmark(
        sampleSize: number = 10
    ): Promise<number> {
        var samples = [];
        const noiseBuffer = new Float32Array(16000);
        for (var i = 0; i < sampleSize; i++) {
            // fill the buffer with noise
            for (let j = 0; j < length; j++) {
                noiseBuffer[j] = Math.random() * 2 - 1;
            }
            await this.generate(noiseBuffer);
            samples.push(this.lastLatency);
        }
        return samples.reduce((sum, num) => sum + num, 0) / sampleSize;
    }

    /**
     * Returns the latency (in ms) of the most recent call to {@link MoonshineModel.generate}
     * 
     * @returns A latency reading (in ms)
     */
    public getLatency(): number {
        return this.lastLatency;
    }

    /**
     * Load the model weights.
     *
     * @remarks This can be a somewhat long-running (in the tens of seconds) async operation, depending on the user's connection and your choice of model (tiny vs base). 
     * To avoid weird async problems that can occur with multiple calls to loadModel, we store and return a single Promise that resolves when the model is loaded.
     */
    public async loadModel(): Promise<void> {
        if (!this.loadPromise) {
            this.loadPromise = this.load();
        }
        return this.loadPromise;
    }

    private async load(): Promise<void> {
        if (!this.isLoading() && !this.isLoaded()) {
            this.isModelLoading = true;
            const sessionOption = MoonshineModel.getSessionOption();
            Log.info(
                `MoonshineModel.loadModel(): Loading model. Using executionProviders: ${sessionOption.executionProviders}`
            );

            this.model.encoder = await ort.InferenceSession.create(
                this.modelURL + "/" + this.precision + "/encoder_model.ort",
                sessionOption
            );

            this.model.decoder = await ort.InferenceSession.create(
                this.modelURL +
                "/" +
                this.precision +
                "/decoder_model_merged.ort",
                sessionOption
            );
            this.isModelLoading = false;
        } else {
            Log.log(
                `MoonshineModel.loadModel(): Ignoring duplicate call. isLoading = ${this.isLoading()} and isLoaded = ${this.isLoaded()}`
            );
        }
    }

    /**
     * Returns whether or not the model is in the process of loading.
     * 
     * @returns `true` if the model is currently loading, `false` if not.
     */
    public isLoading(): boolean {
        return this.isModelLoading;
    }

    /**
     * Returns whether or not the model weights have been loaded.
     * 
     * @returns `true` if the model is loaded, `false` if not.
     */
    public isLoaded(): boolean {
        return (
            this.model.encoder !== undefined && this.model.decoder !== undefined
        );
    }

    /**
     * Generate a transcription of the passed audio.
     *
     * @param audio A `Float32Array` containing raw audio samples from an audio source (e.g., a wav file, or a user's microphone)
     * @returns A `Promise<string>` that resolves with the generated transcription.
     */
    public async generate(audio: Float32Array): Promise<string> {
        const result = await this.generateWithTimestamps(audio);
        return result?.text;
    }

    /**
     * Generate a transcription with optional word-level timestamps.
     *
     * If the decoder model has cross_attentions outputs (decoder_with_attention),
     * word timestamps are computed via DTW alignment. Otherwise, only text is returned.
     *
     * @param audio A `Float32Array` containing raw audio samples
     * @returns A `Promise` resolving with text and optional word timings.
     */
    public async generateWithTimestamps(audio: Float32Array): Promise<{ text: string; words?: WordTiming[] } | undefined> {
        if (this.isLoaded()) {
            const t0 = performance.now();
            const maxLen = Math.trunc(audio.length / 16000 * 14);
            const audioDuration = audio.length / 16000;

            var encoderInput = {
                input_values: new ort.Tensor("float32", audio, [
                    1,
                    audio.length,
                ]),
            };
            var encoderAttentionMask = undefined;
            if (this.model.encoder.inputNames.includes("attention_mask")) {
                var maskData = new BigInt64Array(audio.length);
                maskData.fill(BigInt(1));
                encoderAttentionMask = new ort.Tensor("int64", maskData, [
                    1,
                    audio.length,
                ]);
                Object.assign(encoderInput, {
                    attention_mask: encoderAttentionMask,
                });
            }

            const encoderOutput = await this.model.encoder.run(encoderInput);

            var pastKeyValues = Object.fromEntries(
                Array.from({ length: this.shape.numLayers }, (_, i) =>
                    ["decoder", "encoder"].flatMap((a) =>
                        ["key", "value"].map((b) => [
                            `past_key_values.${i}.${a}.${b}`,
                            new ort.Tensor(
                                "float32",
                                [],
                                [
                                    0,
                                    this.shape.numKVHeads,
                                    1,
                                    this.shape.headDim,
                                ]
                            ),
                        ])
                    )
                ).flat()
            );

            // Check if decoder has cross-attention outputs
            const hasAttention = this.model.decoder.outputNames.some(
                (name: string) => name.startsWith("cross_attentions.")
            );
            const attnOutputNames = hasAttention
                ? this.model.decoder.outputNames.filter((name: string) =>
                      name.startsWith("cross_attentions.")
                  )
                : [];
            const crossAttentionSteps: Float32Array[] = [];

            var tokens = [this.decoderStartTokenID];
            var inputIDs = [tokens];

            for (let i = 0; i < maxLen; i++) {
                var decoderInput = {
                    // @ts-expect-error
                    input_ids: new ort.Tensor("int64", inputIDs, [
                        1,
                        inputIDs.length,
                    ]),
                    encoder_hidden_states: encoderOutput.last_hidden_state,
                    use_cache_branch: new ort.Tensor("bool", [i > 0]),
                };
                if (encoderAttentionMask) {
                    Object.assign(decoderInput, {
                        encoder_attention_mask: encoderAttentionMask,
                    });
                }
                Object.assign(decoderInput, pastKeyValues);
                var decoderOutput = await this.model.decoder.run(decoderInput);

                var logits = await decoderOutput.logits.getData();
                var nextToken = argMax(logits);
                tokens.push(nextToken);

                // Collect cross-attention if available
                if (hasAttention) {
                    for (const attnName of attnOutputNames) {
                        const attnTensor = decoderOutput[attnName];
                        if (attnTensor) {
                            const attnData = await attnTensor.getData();
                            crossAttentionSteps.push(new Float32Array(attnData));
                        }
                    }
                }

                if (nextToken == this.eosTokenID) break;
                inputIDs = [[nextToken]];

                const presentKeyValues = Object.entries(decoderOutput)
                    .filter(([key, _]) => key.includes("present"))
                    .map(([_, value]) => value);

                Object.keys(pastKeyValues).forEach((k, index) => {
                    const v = presentKeyValues[index];
                    if (!(i > 0) || k.includes("decoder")) {
                        pastKeyValues[k] = v;
                    }
                });
            }
            this.lastLatency = performance.now() - t0;
            const text = llamaTokenizer.decode(tokens.slice(0, -1));

            // Build word timestamps if attention was collected
            let words: WordTiming[] | undefined = undefined;
            if (crossAttentionSteps.length > 0 && tokens.length > 2) {
                const numLayers = attnOutputNames.length;
                const numSteps = crossAttentionSteps.length / numLayers;
                // Each step has numLayers entries, each [1, heads, 1, encFrames]
                // Get dimensions from first entry
                const firstStep = crossAttentionSteps[0];
                const numHeads = this.shape.numKVHeads;
                const encFrames = firstStep.length / numHeads; // [heads * encFrames]

                // Rearrange from [step0_L0, step0_L1, ..., step1_L0, ...]
                // to [layers*heads, totalSteps, encFrames]
                const totalSize = numLayers * numHeads * numSteps * encFrames;
                const rearranged = new Float32Array(totalSize);
                for (let s = 0; s < numSteps; s++) {
                    for (let l = 0; l < numLayers; l++) {
                        const src = crossAttentionSteps[s * numLayers + l];
                        for (let h = 0; h < numHeads; h++) {
                            const dstOffset = ((l * numHeads + h) * numSteps + s) * encFrames;
                            const srcOffset = h * encFrames;
                            for (let e = 0; e < encFrames; e++) {
                                rearranged[dstOffset + e] = src[srcOffset + e];
                            }
                        }
                    }
                }

                words = buildWordTimings(
                    rearranged,
                    numLayers,
                    numHeads,
                    numSteps,
                    encFrames,
                    tokens,
                    audioDuration,
                    (id: number) => llamaTokenizer.decode([id]),
                    (id: number) => {
                        // Get raw token for word boundary detection
                        // llama-tokenizer-js vocab uses ▁ prefix for word starts
                        const decoded = llamaTokenizer.decode([id]);
                        // Check if original vocab entry starts with ▁
                        const vocab = llamaTokenizer.vocab;
                        if (vocab && id < vocab.length) {
                            return vocab[id] || decoded;
                        }
                        return decoded;
                    }
                );
            }

            return { text, words };
        } else {
            Log.warn(
                "MoonshineModel.generate(): Tried to call generate before the model was loaded."
            );
        }
        return undefined;
    }
}
