
const input = [
    [0.2, 0.5, 0.1, 0.3],
    [0.4, 0.1, 0.6, 0.2],
    [0.3, 0.7, 0.2, 0.5]
];

// 1. Initialize weight matrices

export function initializeWeights(inputDim, outputDim) {
    const weights = [];

    for (let i = 0; i < inputDim; i++) {
        const row = [];

        for (let j = 0; j < outputDim; j++) {
            const randomValue = Math.random() * 2 - 1;
            row.push(randomValue);
        }

        weights.push(row);
    }

    return weights;
}

const embeddingDim = 4;

const WQ = initializeWeights(embeddingDim, embeddingDim);
const WK = initializeWeights(embeddingDim, embeddingDim);
const WV = initializeWeights(embeddingDim, embeddingDim);
const WO = initializeWeights(embeddingDim, embeddingDim);


// 2. Matrix multiplication

export function matrixMultiply(A, B) {
    if (A[0].length !== B.length) {
        throw new Error("Incompatible matrix dimensions");
    }

    const result = [];

    for (let i = 0; i < A.length; i++) {
        const row = [];

        for (let j = 0; j < B[0].length; j++) {
            let sum = 0;

            for (let k = 0; k < B.length; k++) {
                sum += A[i][k] * B[k][j];
            }

            row.push(sum);
        }

        result.push(row);
    }

    return result;
}


// 3. Generate Q, K, V

const Q = matrixMultiply(input, WQ);
const K = matrixMultiply(input, WK);
const V = matrixMultiply(input, WV);

console.log("Query:", Q);
console.log("Key:", K);
console.log("Value:", V);


// 4. Transpose matrix

export function transpose(matrix) {
    const result = [];

    for (let col = 0; col < matrix[0].length; col++) {
        const row = [];

        for (let r = 0; r < matrix.length; r++) {
            row.push(matrix[r][col]);
        }

        result.push(row);
    }

    return result;
}


// 5. Calculate attention scores: Q × K transpose

export function calculateAttentionScores(Q, K) {
    const transposedK = transpose(K);

    return matrixMultiply(Q, transposedK);
}

const scores = calculateAttentionScores(Q, K);

console.log("Attention Scores:", scores);


// 6. Scale attention scores

export function scaleAttentionScores(scores, keyDim) {
    const scale = Math.sqrt(keyDim);

    return scores.map(row =>
        row.map(value => value / scale)
    );
}

const scaledScores = scaleAttentionScores(scores, embeddingDim);

console.log("Scaled Scores:", scaledScores);


// 7. Apply causal masking

function applyCausalMask(scores) {
    const maskedScores = scores.map(row => [...row]);

    for (let i = 0; i < maskedScores.length; i++) {
        for (let j = i + 1; j < maskedScores[i].length; j++) {
            maskedScores[i][j] = -Infinity;
        }
    }

    return maskedScores;
}

const maskedScores = applyCausalMask(scaledScores);

console.log("Masked Scores:", maskedScores);


// 8. Softmax

function softmax(row) {
    const maxScore = Math.max(...row);

    const exponentials = row.map(score =>
        Math.exp(score - maxScore)
    );

    const total = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    return exponentials.map(value => value / total);
}

export function softmaxMatrix(scores) {
    return scores.map(row => softmax(row));
}

const attentionWeights = softmaxMatrix(maskedScores);

console.log("Attention Weights:", attentionWeights);


// 9. Calculate attention output: Attention × V

export function calculateAttentionOutput(attentionWeights, V) {
    return matrixMultiply(attentionWeights, V);
}

const attentionOutput = calculateAttentionOutput(
    attentionWeights,
    V
);

console.log("Attention Output:", attentionOutput);


// 10. Output projection: Attention Output × WO

const projectedOutput = matrixMultiply(attentionOutput, WO);

console.log("Projected Output:", projectedOutput);


