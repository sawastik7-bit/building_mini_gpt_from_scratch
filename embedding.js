


function initializeEmbeddings(vocab_size,embedding_dim){
    const embeddings=[];

    for(let i=0;i<vocab_size;i++){
        const vector=[];
        for(let j=0;j<embedding_dim;j++){
 const val=Math.random() * 2 -1;
 vector.push(val);

        }

        embeddings.push(vector);
    }

return embeddings;
}




function initializePositionEmbeddings(sequenceLength, embeddingDim) {
    const positionEmbeddings = [];

    for (let i = 0; i < sequenceLength; i++) {
        const positionVector = [];

        for (let j = 0; j < embeddingDim; j++) {
            const randomValue = Math.random() * 2 - 1;
            positionVector.push(randomValue);
        }

        positionEmbeddings.push(positionVector);
    }

    return positionEmbeddings;
}


function embeddingForward(tokenIds, embeddings) {  // searach specific embeddings present at specific row 
    const output = [];

    for (let tokenId of tokenIds) {
        // Check whether the token ID exists in the matrix
        if (
            !Number.isInteger(tokenId) ||
            tokenId < 0 ||
            tokenId >= embeddings.length
        ) {
            throw new Error(`Invalid token ID: ${tokenId}`);
        }

        // Retrieve the vector belonging to this token
        const vector = embeddings[tokenId];

        output.push([...vector]);
    }

    return output;
}



function addPositionEmbeddings(tokenEmbeddings, positionEmbeddings) {
    if (tokenEmbeddings.length !== positionEmbeddings.length) {
        throw new Error("Token and position counts must match");
    }

    const output = [];

    for (let i = 0; i < tokenEmbeddings.length; i++) {
        if (tokenEmbeddings[i].length !== positionEmbeddings[i].length) {
            throw new Error("Embedding dimensions must match");
        }

        const combinedVector = [];

        for (let j = 0; j < tokenEmbeddings[i].length; j++) {
            const value =
                tokenEmbeddings[i][j] + positionEmbeddings[i][j];

            combinedVector.push(value);
        }

        output.push(combinedVector);
    }

    return output;
}



