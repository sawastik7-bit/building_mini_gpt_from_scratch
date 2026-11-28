
import { initializeWeights } from "./selfAttention.js";

import {
    initializeTransformerBlockParams,
    transformerBlock
} from "./transformerBlock.js";

import {
    initializeFinalLayerNormParams,
    finalLayerNorm
} from "./finalLayerNorm.js";

import {
    initializeLMHeadParams,
    lmHeadForward
} from "./lmHead.js";

import { softmaxMatrix } from "./softmax.js";


// =====================================
// 1. ADD TWO MATRICES
// =====================================

function addMatrices(A, B) {

    if (A.length !== B.length) {
        throw new Error("Matrices must have the same number of rows");
    }

    return A.map((row, i) => {

        if (row.length !== B[i].length) {
            throw new Error("Matrices must have the same number of columns");
        }

        return row.map((value, j) => value + B[i][j]);

    });

}


// =====================================
// 2. TOKEN EMBEDDING LOOKUP
// =====================================

function getTokenEmbeddings(tokenIds, embeddingTable) {

    return tokenIds.map(tokenId => {

        if (
            !Number.isInteger(tokenId) ||
            tokenId < 0 ||
            tokenId >= embeddingTable.length
        ) {
            throw new Error(`Invalid token ID: ${tokenId}`);
        }

        return [...embeddingTable[tokenId]];

    });

}


// =====================================
// 3. POSITION EMBEDDING LOOKUP
// =====================================

function getPositionEmbeddings(sequenceLength, positionTable) {

    if (sequenceLength > positionTable.length) {
        throw new Error("Sequence length exceeds maximum context length");
    }

    return positionTable
        .slice(0, sequenceLength)
        .map(row => [...row]);

}


// =====================================
// 4. INITIALIZE GPT
// =====================================

export function initializeGPT({
    vocabSize,
    modelDim,
    maxSequenceLength,
    numLayers,
    numHeads,
    hiddenDim
}) {

    if (modelDim % numHeads !== 0) {
        throw new Error("modelDim must be divisible by numHeads");
    }

    // Token embedding table: vocabSize x modelDim

    const tokenEmbeddings = initializeWeights(
        vocabSize,
        modelDim
    );


    // Position embedding table: maxSequenceLength x modelDim

    const positionEmbeddings = initializeWeights(
        maxSequenceLength,
        modelDim
    );


    // Initialize Transformer Blocks

    const transformerBlocks = [];

    for (let i = 0; i < numLayers; i++) {

        const blockParams = initializeTransformerBlockParams(
            modelDim,
            numHeads,
            hiddenDim
        );

        transformerBlocks.push(blockParams);

    }


    // Initialize final LayerNorm

    const finalLN = initializeFinalLayerNormParams(modelDim);


    // Initialize Language Model Head

    const lmHead = initializeLMHeadParams(
        modelDim,
        vocabSize
    );


    return {
        vocabSize,
        modelDim,
        maxSequenceLength,
        numLayers,
        numHeads,
        hiddenDim,

        tokenEmbeddings,
        positionEmbeddings,

        transformerBlocks,

        finalLN,
        lmHead
    };

}


// =====================================
// 5. GPT FORWARD PASS
// =====================================

export function gptForward(tokenIds, params) {

    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
        throw new Error("tokenIds must be a non-empty array");
    }

    if (tokenIds.length > params.maxSequenceLength) {
        throw new Error("Input exceeds maximum sequence length");
    }


    // STEP 1: Token Embeddings

    const tokenVectors = getTokenEmbeddings(
        tokenIds,
        params.tokenEmbeddings
    );


    // STEP 2: Position Embeddings

    const positionVectors = getPositionEmbeddings(
        tokenIds.length,
        params.positionEmbeddings
    );


    // STEP 3: Add Token + Position Embeddings

    let output = addMatrices(
        tokenVectors,
        positionVectors
    );


    // STEP 4: Pass through all Transformer Blocks

    for (const blockParams of params.transformerBlocks) {

        output = transformerBlock(
            output,
            blockParams
        );

    }


    // STEP 5: Final LayerNorm

    const normalizedOutput = finalLayerNorm(
        output,
        params.finalLN
    );


    // STEP 6: Language Model Head

    const logits = lmHeadForward(
        normalizedOutput,
        params.lmHead
    );


    // STEP 7: Convert logits into probabilities

    const probabilities = softmaxMatrix(logits);


    // Return all useful outputs

    return {
        hiddenStates: output,
        normalizedOutput,
        logits,
        probabilities
    };

}
