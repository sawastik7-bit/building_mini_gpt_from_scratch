import {
    initializeLayerNormParams,
    layerNorm
} from "./layerNorm.js";

import {
    initializeMultiHeadAttentionParams,
    multiHeadAttention
} from "./multiHeadAttention.js";

import {
    initializeFFNParams,
    feedForward
} from "./feedForward.js";

import { residualConnection } from "./residual.js";


export function initializeTransformerBlockParams(
    modelDim,
    numHeads,
    hiddenDim
) {

    return {

        // First LayerNorm
        ln1: initializeLayerNormParams(modelDim),

        // Multi-head attention
        attention: initializeMultiHeadAttentionParams(
            modelDim,
            numHeads
        ),

        // Second LayerNorm
        ln2: initializeLayerNormParams(modelDim),

        // Feed Forward Network
        ffn: initializeFFNParams(
            modelDim,
            hiddenDim
        )
    };
}


export function transformerBlock(input, params) {

    // STEP 1: First LayerNorm
    const normalizedInput = layerNorm(
        input,
        params.ln1.gamma,
        params.ln1.beta
    );

    // STEP 2: Multi-Head Attention
    const attentionOutput = multiHeadAttention(
        normalizedInput,
        params.attention
    );

    // STEP 3: First Residual Connection
    const residual1 = residualConnection(
        input,
        attentionOutput
    );

    // STEP 4: Second LayerNorm
    const normalizedResidual = layerNorm(
        residual1,
        params.ln2.gamma,
        params.ln2.beta
    );

    // STEP 5: Feed Forward Network
    const ffnOutput = feedForward(
        normalizedResidual,
        params.ffn
    );

    // STEP 6: Second Residual Connection
    const finalOutput = residualConnection(
        residual1,
        ffnOutput
    );

    return finalOutput;
}