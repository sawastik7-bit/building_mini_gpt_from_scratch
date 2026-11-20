
let vocabulary = new Set();
let merges = [];

function tokenize(text) {
    if (!text.trim()) {
        console.log("There is no textual data to split");
        return [];
    }

    let words = text.trim().split(/\s+/);
    let tokenized_array = [];

    for (let word of words) {
        let arr = [];

        for (let char of word) {
            arr.push(char);
        }

        tokenized_array.push(arr);
    }

    return tokenized_array;
}

function countPairs(words) {
    let pairCounts = new Map();

    for (let word of words) {
        for (let i = 0; i < word.length - 1; i++) {
            let left = word[i];
            let right = word[i + 1];

            let pairKey = JSON.stringify([left, right]);

            pairCounts.set(
                pairKey,
                (pairCounts.get(pairKey) || 0) + 1
            );
        }
    }

    return pairCounts;
}

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

function mergePair(tokenized_text_array, pair) {
    let final_merged_array = [];

    for (let word of tokenized_text_array) {
        let merged_pair_array = [];

        for (let i = 0; i < word.length; i++) {
            if (
                i < word.length - 1 &&
                word[i] === pair.left &&
                word[i + 1] === pair.right
            ) {
                merged_pair_array.push(pair.result);
                i++;
            } else {
                merged_pair_array.push(word[i]);
            }
        }

        final_merged_array.push(merged_pair_array);
    }

    return final_merged_array;
}

function train(raw_text_string, number_of_iterations) {
    vocabulary.clear();
    merges = [];

    let splitted_text_array = tokenize(raw_text_string);

    for (let word of splitted_text_array) {
        for (let token of word) {
            vocabulary.add(token);
        }
    }

    for (let i = 0; i < number_of_iterations; i++) {
        let pair_count = countPairs(splitted_text_array);
        let best_pair = getMostFrequentPair(pair_count);

        if (!best_pair) break;

        splitted_text_array = mergePair(
            splitted_text_array,
            best_pair
        );

        vocabulary.add(best_pair.result);
        merges.push(best_pair);

        console.log(
            `Iteration ${i + 1}: ${best_pair.left} + ${best_pair.right} = ${best_pair.result}`
        );

        console.log(splitted_text_array);
    }

    return splitted_text_array;
}

function createTokenIDs(vocabulary) {
    let tokenToId = new Map();
    let id = 0;

    for (let token of vocabulary) {
        tokenToId.set(token, id);
        id++;
    }

    return tokenToId;
}

function encode(text, tokenToId) {
    let tokenized_array = tokenize(text);
    let encoded_words = [];

    for (let word of tokenized_array) {
        let tokens = word;

        for (let merge of merges) {
            let merged_word = [];

            for (let i = 0; i < tokens.length; i++) {
                if (
                    i < tokens.length - 1 &&
                    tokens[i] === merge.left &&
                    tokens[i + 1] === merge.right
                ) {
                    merged_word.push(merge.result);
                    i++;
                } else {
                    merged_word.push(tokens[i]);
                }
            }

            tokens = merged_word;
        }

        let token_ids = [];

        for (let token of tokens) {
            if (!tokenToId.has(token)) {
                throw new Error(`Unknown token: ${token}`);
            }

            token_ids.push(tokenToId.get(token));
        }

        encoded_words.push(token_ids);
    }

    return encoded_words;
}

let raw_text_string = "low lower lowest";
let number_of_iterations = 10;

let final_trained_data = train(
    raw_text_string,
    number_of_iterations
);

let tokenToId = createTokenIDs(vocabulary);

console.log("Final trained data:", final_trained_data);
console.log("Vocabulary:", [...vocabulary]);
console.log("Token IDs:", [...tokenToId.entries()]);
console.log("Merge rules:", merges);
console.log("Vocabulary size:", vocabulary.size);
console.log("Total merges:", merges.length);

let new_sentence = "low lowest";
let encoded_result = encode(new_sentence, tokenToId);

console.log("New sentence:", new_sentence);
console.log("Encoded result:", encoded_result);
