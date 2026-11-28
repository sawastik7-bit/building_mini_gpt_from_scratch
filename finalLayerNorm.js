
import {
    initializeLayerNormParams,
    layerNorm
} from "./layerNorm.js";


// Initialize final LayerNorm parameters

export function initializeFinalLayerNormParams(modelDim) {

    return initializeLayerNormParams(modelDim);

}


// Apply final LayerNorm

export function finalLayerNorm(input, params) {

    return layerNorm(
        input,
        params.gamma,
        params.beta
    );

}
