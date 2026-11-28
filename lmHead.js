
import {
    initializeWeights,
    matrixMultiply
} from "./selfAttention.js";


// Initialize Language Model Head parameters

export function initializeLMHeadParams(modelDim, vocabSize) {

    return {
        W: initializeWeights(modelDim, vocabSize)
    };

}


// Convert hidden representations into vocabulary logits

export function lmHeadForward(input, params) {

    const logits = matrixMultiply(
        input,
        params.W
    );

    return logits;

}
