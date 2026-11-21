

const input = [   // 3 input token with 4 dimensions 
    [0.2, 0.5, 0.1, 0.3],
    [0.4, 0.1, 0.6, 0.2],
    [0.3, 0.7, 0.2, 0.5]
];


// initialize the weight matrices WQ, WK, and WV


function initializeWeights(inputDim, outputDim) {
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



function matrixMultiply(A, B) {
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


const Q = matrixMultiply(input, WQ);
const K = matrixMultiply(input, WK);
const V = matrixMultiply(input, WV);

console.log("Query:", Q);
console.log("Key:", K);
console.log("Value:", V);

function transpose(matrix) {
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



function calculateAttentionScores(Q, K) {
    const transposedK = transpose(K);

    const scores = matrixMultiply(Q, transposedK);

    return scores;
}

let scores=calculateAttentionScores(Q,K);


function scaleAttentionScores(scores, keyDim) {   // decreasing large positive values so that its easy to apply soft max
    const scale = Math.sqrt(keyDim);

    return scores.map(row =>
        row.map(value => value / scale)
    );
}


 
function applyCausalMask(scores) {     // during training phase no token should be able to see its future tokens 

    const maskedScores = scores.map(row => [...row]);

    for (let i = 0; i < maskedScores.length; i++) {
        for (let j = i + 1; j < maskedScores[i].length; j++) {
            maskedScores[i][j] = -Infinity;
        }
    }

    return maskedScores;
}


// converting scores to probabilities

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



function softmaxMatrix(scores) {
    return scores.map(row => softmax(row));
}


// now the weighted sum to find values 

function calculateAttentionOutput(attentionWeights, V) {
    const output = matrixMultiply(attentionWeights, V);

    return output;
}


const attentionOutput = calculateAttentionOutput(
    attentionWeights,
    V
);

console.log(attentionOutput);