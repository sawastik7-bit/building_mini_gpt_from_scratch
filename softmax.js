
export function softmax(logits) {

    const maxLogit = Math.max(...logits);

    // Calculate exponentials after subtracting max

    const exponentials = logits.map(value =>
        Math.exp(value - maxLogit)
    );

    // Calculate sum of exponentials

    const sum = exponentials.reduce(
        (total, value) => total + value,
        0
    );

    // Convert into probabilities

    return exponentials.map(value =>
        value / sum
    );

}


// Apply Softmax separately to every token position

export function softmaxMatrix(matrix) {

    return matrix.map(row => softmax(row));

}
