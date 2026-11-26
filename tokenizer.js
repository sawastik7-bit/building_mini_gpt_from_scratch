
let vocabulary = new Set();
let merges = [];

// STEP 1: Convert text into UTF-8 bytes represented in hexadecimal.
function tokenize(text) {
    if (!text.length) {
        console.log("There is no textual data to split");
        return [];
    }

    const bytes = Buffer.from(text, "utf8");

    let tokens = [];

    for (let byte of bytes) {
        let hexToken = byte.toString(16).padStart(2, "0");
        tokens.push(hexToken);
    }

    return tokens;
}


// STEP 2: Count adjacent token pairs.
function countPairs(tokenizedText) {
    let pairCounts = new Map();

    // We now have ONE continuous byte sequence.
    for (let i = 0; i < tokenizedText.length - 1; i++) {
        let left = tokenizedText[i];
        let right = tokenizedText[i + 1];

        let pairKey = JSON.stringify([left, right]);

        pairCounts.set(
            pairKey,
            (pairCounts.get(pairKey) || 0) + 1
        );
    }

    return pairCounts;
}


// STEP 3: Find the most frequent adjacent pair.
function getMostFrequentPair(pairCounts) {
    let bestPair = null;
    let maxCount = 0;

    for (let [pairKey, count] of pairCounts) {
        if (count > maxCount) {
            maxCount = count;

            let [left, right] = JSON.parse(pairKey);

            bestPair = {
                left,
                right,
                result: left + right
            };
        }
    }

    return bestPair;
}


// STEP 4: Merge the selected pair throughout the sequence.
function mergePair(tokenizedText, pair) {
    let mergedTokens = [];

    for (let i = 0; i < tokenizedText.length; i++) {

        if (
            i < tokenizedText.length - 1 &&
            tokenizedText[i] === pair.left &&
            tokenizedText[i + 1] === pair.right
        ) {
            mergedTokens.push(pair.result);
            i++;
        } else {
            mergedTokens.push(tokenizedText[i]);
        }
    }

    return mergedTokens;
}


// STEP 5: Initialize all 256 possible byte tokens.
function initializeByteVocabulary() {
    vocabulary.clear();

    for (let byte = 0; byte < 256; byte++) {
        let hexToken = byte.toString(16).padStart(2, "0");
        vocabulary.add(hexToken);
    }
}


// STEP 6: Train BPE.
function train(rawText, numberOfIterations) {
    initializeByteVocabulary();
    merges = [];

    let tokenizedText = tokenize(rawText);

    for (let i = 0; i < numberOfIterations; i++) {

        let pairCounts = countPairs(tokenizedText);

        let bestPair = getMostFrequentPair(pairCounts);

        if (!bestPair) break;

        tokenizedText = mergePair(tokenizedText, bestPair);

        vocabulary.add(bestPair.result);
        merges.push(bestPair);

        console.log(
            `Iteration ${i + 1}: ${bestPair.left} + ${bestPair.right} = ${bestPair.result}`
        );

        console.log("Current tokens:", tokenizedText);
    }

    return tokenizedText;
}


// STEP 7: Assign token IDs.
function createTokenIDs(vocabulary) {
    let tokenToId = new Map();

    // Original byte tokens get IDs 0–255.
    for (let byte = 0; byte < 256; byte++) {
        let hexToken = byte.toString(16).padStart(2, "0");
        tokenToId.set(hexToken, byte);
    }

    // Newly merged tokens get IDs starting from 256.
    let id = 256;

    for (let token of vocabulary) {
        if (!tokenToId.has(token)) {
            tokenToId.set(token, id);
            id++;
        }
    }

    return tokenToId;
}


// STEP 8: Encode new text using the learned merges.
function encode(text, tokenToId) {
    let tokens = tokenize(text);

    // Apply learned merge rules in their training order.
    for (let merge of merges) {
        let mergedTokens = [];

        for (let i = 0; i < tokens.length; i++) {

            if (
                i < tokens.length - 1 &&
                tokens[i] === merge.left &&
                tokens[i + 1] === merge.right
            ) {
                mergedTokens.push(merge.result);
                i++;
            } else {
                mergedTokens.push(tokens[i]);
            }
        }

        tokens = mergedTokens;
    }

    // Convert final tokens into integer IDs.
    let tokenIds = [];

    for (let token of tokens) {
        if (!tokenToId.has(token)) {
            throw new Error(`Unknown token: ${token}`);
        }

        tokenIds.push(tokenToId.get(token));
    }

    return tokenIds;
}


// MAIN PROGRAM

let raw_text_string = "low lower lowest";
let number_of_iterations = 10;

let final_trained_data = train(
    raw_text_string,
    number_of_iterations
);

let tokenToId = createTokenIDs(vocabulary);

console.log("Final trained data:", final_trained_data);
console.log("Vocabulary size:", vocabulary.size);
console.log("Token IDs:", [...tokenToId.entries()]);
console.log("Merge rules:", merges);
console.log("Total merges:", merges.length);

let new_sentence = "low lowest";

let encoded_result = encode(new_sentence, tokenToId);

console.log("New sentence:", new_sentence);
console.log("Encoded result:", encoded_result);
