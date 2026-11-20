
let vocabulary = new Set(); // Stores all tokens
let merges = [];            // Stores learned merge rules


// 1. TOKENIZE
function tokenize(text) {

    if (!text.trim()) {
        console.log("There is no textual data to split");
        return [];
    }

    let words = text.trim().split(/\s+/);
    let tokenized_array = [];

    for (let word of words) {
        let arr = [];

        for (let i = 0; i < word.length; i++) {
            arr.push(word[i]);
        }

        tokenized_array.push(arr);
    }

    return tokenized_array;
}


// 2. COUNT ADJACENT PAIRS
function countPairs(words) {

    let pairCounts = new Map();

    for (let word of words) {

        for (let i = 0; i < word.length - 1; i++) {

            let left = word[i];
            let right = word[i + 1];

            // Unique key for each token pair
            let pairKey = JSON.stringify([left, right]);

            pairCounts.set(
                pairKey,
                (pairCounts.get(pairKey) || 0) + 1
            );
        }
    }

    return pairCounts;
}


// 3. FIND MOST FREQUENT PAIR
function getMostFrequentPair(pairCounts) {

    let bestPair = null;
    let maxCount = 0;

    for (let [pairKey, count] of pairCounts) {

        if (count > maxCount) {

            maxCount = count;

            let [left, right] = JSON.parse(pairKey);

            bestPair = {
                left: left,
                right: right,
                result: left + right
            };
        }
    }

    return bestPair;
}


// 4. MERGE THE SELECTED PAIR
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

                i++; // Skip the next token because it was merged

            } else {

                merged_pair_array.push(word[i]);

            }
        }

        final_merged_array.push(merged_pair_array);
    }

    return final_merged_array;
}


// 5. TRAIN THE TOKENIZER
function train(raw_text_string, number_of_iterations) {

    // Reset previous training data
    vocabulary.clear();
    merges = [];

    // Initial tokenization
    let splitted_text_array = tokenize(raw_text_string);

    // Add original character tokens to vocabulary
    for (let word of splitted_text_array) {
        for (let token of word) {
            vocabulary.add(token);
        }
    }

    console.log("Initial tokens:", splitted_text_array);
    console.log("Initial vocabulary:", [...vocabulary]);

    // BPE training loop
    for (let i = 0; i < number_of_iterations; i++) {

        // 1. Count adjacent pairs
        let pair_count = countPairs(splitted_text_array);

        // 2. Find most frequent pair
        let best_pair = getMostFrequentPair(pair_count);

        // 3. Stop if no pairs exist
        if (!best_pair) {
            console.log("No more pairs available.");
            break;
        }

        // 4. Merge the selected pair
        splitted_text_array = mergePair(
            splitted_text_array,
            best_pair
        );

        // 5. Add newly created token to vocabulary
        vocabulary.add(best_pair.result);

        // 6. Store the complete merge rule
        merges.push(best_pair);

        console.log(`\nIteration ${i + 1}:`);
        console.log(
            `${best_pair.left} + ${best_pair.right} = ${best_pair.result}`
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



// 6. RUN TRAINING
let raw_text_string = "low lower lowest";
let number_of_iterations = 10;

let final_trained_data = train(
    raw_text_string,
    number_of_iterations
);

let tokenToId = createTokenIDs(vocabulary);

console.log([...tokenToId.entries()]);

// 7. DISPLAY RESULTS
console.log("\n========== TRAINING COMPLETE ==========");

console.log("Final trained data:");
console.log(final_trained_data);

console.log("\nComplete vocabulary:");
console.log([...vocabulary]);

console.log("\nLearned merge rules:");
console.log(merges);

console.log("\nTotal vocabulary size:", vocabulary.size);
console.log("Total merges learned:", merges.length);
