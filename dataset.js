
function createSamples(tokenIds, sequenceLength) {
    const samples = [];

    for (let i = 0; i + sequenceLength < tokenIds.length; i++) {
        const input = tokenIds.slice(i, i + sequenceLength);

        const target = tokenIds.slice(
            i + 1,
            i + sequenceLength + 1
        );

        samples.push({ input, target });
    }

    return samples;
}

function createBatches(samples, batchSize) {
    const batches = [];

    for (let i = 0; i < samples.length; i += batchSize) {
        const batch = samples.slice(i, i + batchSize);

        batches.push({
            inputs: batch.map(sample => sample.input),
            targets: batch.map(sample => sample.target)
        });
    }

    return batches;
}

function loadDataset(tokenIds, sequenceLength, batchSize) {
    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
        throw new Error("Invalid or empty token IDs");
    }

    if (!Number.isInteger(sequenceLength) || sequenceLength < 1) {
        throw new Error("Invalid sequence length");
    }

    if (!Number.isInteger(batchSize) || batchSize < 1) {
        throw new Error("Invalid batch size");
    }

    const samples = createSamples(tokenIds, sequenceLength);
    const batches = createBatches(samples, batchSize);

    return {
        samples,
        batches,
        totalSamples: samples.length,
        totalBatches: batches.length
    };
}

export { createSamples, createBatches, loadDataset };
