/**
 * Word-level timestamp alignment using Dynamic Time Warping (DTW)
 * on cross-attention weights from the Moonshine decoder.
 */

export interface WordTiming {
    word: string;
    start: number;
    end: number;
    confidence: number;
}

/**
 * Dynamic Time Warping on a cost matrix [N x M].
 * Returns aligned (textIndices, timeIndices) arrays.
 */
function dtw(
    costMatrix: Float32Array,
    N: number,
    M: number
): { textIndices: Int32Array; timeIndices: Int32Array } {
    // Cumulative cost matrix D[N+1][M+1], init to Infinity
    const D = new Float64Array((N + 1) * (M + 1)).fill(Infinity);
    D[0] = 0.0;
    // Trace matrix for backtracking: 0=diagonal, 1=up, 2=left
    const trace = new Int32Array(N * M);

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            const c0 = D[i * (M + 1) + j]; // diagonal
            const c1 = D[i * (M + 1) + (j + 1)]; // up
            const c2 = D[(i + 1) * (M + 1) + j]; // left
            let argmin = 0;
            let minCost = c0;
            if (c1 < minCost) {
                argmin = 1;
                minCost = c1;
            }
            if (c2 < minCost) {
                argmin = 2;
                minCost = c2;
            }
            trace[i * M + j] = argmin;
            D[(i + 1) * (M + 1) + (j + 1)] = costMatrix[i * M + j] + minCost;
        }
    }

    // Backtrack
    const pathText: number[] = [];
    const pathTime: number[] = [];
    let i = N - 1,
        j = M - 1;
    while (i >= 0 || j >= 0) {
        pathText.push(i);
        pathTime.push(j);
        if (i === 0 && j === 0) break;
        const dir = trace[i * M + j];
        if (dir === 0) {
            i--;
            j--;
        } else if (dir === 1) {
            i--;
        } else {
            j--;
        }
    }
    pathText.reverse();
    pathTime.reverse();
    return {
        textIndices: new Int32Array(pathText),
        timeIndices: new Int32Array(pathTime),
    };
}

/**
 * Apply median filter along the last axis.
 * data is [C, H, W] flattened. Filters along W dimension.
 */
function medianFilter(
    data: Float32Array,
    C: number,
    H: number,
    W: number,
    filterWidth: number
): Float32Array {
    if (filterWidth <= 1) return data;
    if (filterWidth % 2 === 0) filterWidth++;
    const half = Math.floor(filterWidth / 2);
    const result = new Float32Array(data.length);
    const window: number[] = new Array(filterWidth);

    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                // Fill window with reflected padding
                for (let k = 0; k < filterWidth; k++) {
                    let idx = w + k - half;
                    if (idx < 0) idx = -idx;
                    if (idx >= W) idx = 2 * W - idx - 2;
                    idx = Math.max(0, Math.min(W - 1, idx));
                    window[k] = data[(c * H + h) * W + idx];
                }
                // Sort and take median
                window.sort((a, b) => a - b);
                result[(c * H + h) * W + w] =
                    window[Math.floor(filterWidth / 2)];
            }
        }
    }
    return result;
}

/**
 * Check if a token starts a new word using SentencePiece convention.
 * The '▁' character is encoded as UTF-8 bytes 0xE2, 0x96, 0x81.
 */
function tokenStartsNewWord(tokenText: string): boolean {
    return tokenText.startsWith("▁");
}

/**
 * Build word-level timestamps from cross-attention weights collected during decoding.
 *
 * @param crossAttention Flattened [numLayers * numHeads, numTokens, encFrames]
 * @param numLayers Number of decoder layers
 * @param numHeads Number of attention heads per layer
 * @param numTokens Number of decode steps (tokens generated)
 * @param encFrames Number of encoder frames
 * @param tokens Token IDs including BOS and EOS
 * @param audioDuration Duration of the audio in seconds
 * @param decodeToken Function to decode a single token ID to text
 * @param tokenToRaw Function to get the raw token string (for word boundary detection)
 */
export function buildWordTimings(
    crossAttention: Float32Array,
    numLayers: number,
    numHeads: number,
    numTokens: number,
    encFrames: number,
    tokens: number[],
    audioDuration: number,
    decodeToken: (id: number) => string,
    tokenToRaw: (id: number) => string
): WordTiming[] {
    const totalHeads = numLayers * numHeads;

    // Z-score normalize per head along the time axis
    const weights = new Float32Array(crossAttention);
    for (let h = 0; h < totalHeads; h++) {
        for (let t = 0; t < numTokens; t++) {
            const offset = (h * numTokens + t) * encFrames;
            let sum = 0,
                sumSq = 0;
            for (let e = 0; e < encFrames; e++) {
                const v = weights[offset + e];
                sum += v;
                sumSq += v * v;
            }
            const mean = sum / encFrames;
            let std = Math.sqrt(sumSq / encFrames - mean * mean);
            if (std < 1e-10) std = 1e-10;
            for (let e = 0; e < encFrames; e++) {
                weights[offset + e] = (weights[offset + e] - mean) / std;
            }
        }
    }

    // Median filter (width 7)
    const filtered = medianFilter(
        weights,
        totalHeads,
        numTokens,
        encFrames,
        7
    );

    // Average across all heads → [numTokens, encFrames]
    const matrix = new Float32Array(numTokens * encFrames);
    for (let t = 0; t < numTokens; t++) {
        for (let e = 0; e < encFrames; e++) {
            let sum = 0;
            for (let h = 0; h < totalHeads; h++) {
                sum += filtered[(h * numTokens + t) * encFrames + e];
            }
            matrix[t * encFrames + e] = sum / totalHeads;
        }
    }

    // Negate for DTW (DTW minimizes, we want to maximize attention)
    const negMatrix = new Float32Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) negMatrix[i] = -matrix[i];

    // Run DTW
    const { textIndices, timeIndices } = dtw(negMatrix, numTokens, encFrames);

    const timePerFrame = audioDuration / encFrames;

    // Group tokens into words (skip BOS at index 0 and EOS at end)
    const textTokens = tokens.slice(1, -1); // exclude BOS and EOS

    interface WordGroup {
        tokenIds: number[];
        stepIndices: number[];
    }
    const words: WordGroup[] = [];
    let currentTokens: number[] = [];
    let currentSteps: number[] = [];

    for (let i = 0; i < textTokens.length; i++) {
        const stepIdx = i; // step index in DTW matrix (0-based, BOS was step 0 but attention starts from step 0)
        const rawToken = tokenToRaw(textTokens[i]);
        const startsNew = tokenStartsNewWord(rawToken);

        if (startsNew && currentTokens.length > 0) {
            words.push({
                tokenIds: [...currentTokens],
                stepIndices: [...currentSteps],
            });
            currentTokens = [textTokens[i]];
            currentSteps = [stepIdx];
        } else {
            currentTokens.push(textTokens[i]);
            currentSteps.push(stepIdx);
        }
    }
    if (currentTokens.length > 0) {
        words.push({
            tokenIds: [...currentTokens],
            stepIndices: [...currentSteps],
        });
    }

    // Map each word to time via DTW alignment
    const wordTimings: WordTiming[] = [];
    for (const group of words) {
        const wordText = group.tokenIds.map((id) => decodeToken(id)).join("");
        const trimmed = wordText.trim();
        if (!trimmed) continue;

        // Find DTW path points matching this word's step indices
        const stepSet = new Set(group.stepIndices);
        let minFrame = encFrames,
            maxFrame = 0;
        let found = false;
        for (let p = 0; p < textIndices.length; p++) {
            if (stepSet.has(textIndices[p])) {
                minFrame = Math.min(minFrame, timeIndices[p]);
                maxFrame = Math.max(maxFrame, timeIndices[p]);
                found = true;
            }
        }
        if (!found) {
            wordTimings.push({
                word: trimmed,
                start: 0,
                end: 0,
                confidence: 0,
            });
            continue;
        }

        wordTimings.push({
            word: trimmed,
            start: Math.round(minFrame * timePerFrame * 1000) / 1000,
            end: Math.round((maxFrame + 1) * timePerFrame * 1000) / 1000,
            confidence: 1.0,
        });
    }

    // Fix overlapping boundaries
    for (let i = 1; i < wordTimings.length; i++) {
        if (wordTimings[i - 1].end > wordTimings[i].start) {
            const mid =
                (wordTimings[i - 1].end + wordTimings[i].start) / 2;
            wordTimings[i - 1].end = Math.round(mid * 1000) / 1000;
            wordTimings[i].start = Math.round(mid * 1000) / 1000;
        }
    }

    return wordTimings;
}
