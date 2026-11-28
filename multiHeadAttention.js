import {
    initializeWeights,
    matrixMultiply,
    calculateAttentionScores,
    scaleAttentionScores,
    applyCausalMask,
    softmaxMatrix,
    calculateAttentionOutput
} from "./selfAttention";

const WQ = initializeWeights(4, 4);
const WK = initializeWeights(4, 4);
const WV = initializeWeights(4, 4);

const modelDim = 4;
const numHeads = 2;
const headDim = modelDim / numHeads;

// Head 1 weights
const WQ1 = initializeWeights(modelDim, headDim);
const WK1 = initializeWeights(modelDim, headDim);
const WV1 = initializeWeights(modelDim, headDim);

// Head 2 weights
const WQ2 = initializeWeights(modelDim, headDim);
const WK2 = initializeWeights(modelDim, headDim);
const WV2 = initializeWeights(modelDim, headDim);



function singleAttentionHead(input, WQ, WK, WV) {

    // 1. Generate Query, Key and Value
    const Q = matrixMultiply(input, WQ);
    const K = matrixMultiply(input, WK);
    const V = matrixMultiply(input, WV);

    // 2. Calculate attention scores
    const scores = calculateAttentionScores(Q, K);

    // 3. Scale scores
    const scaledScores = scaleAttentionScores(
        scores,
        K[0].length
    );

    // 4. Apply causal mask
    const maskedScores = applyCausalMask(scaledScores);

    // 5. Convert scores into attention weights
    const attentionWeights = softmaxMatrix(maskedScores);

    // 6. Calculate final attention output
    const output = calculateAttentionOutput(
        attentionWeights,
        V
    );

    return output;
}