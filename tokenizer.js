const { Pacifico } = require("next/font/google");

let vocabulary=new Set();  //stores all tokens ,out tokenizer know 
let merges=[];  // stores pairs our tokenizeer learns during training




function tokenize(text){
    let words=text.trim().split(" ");
    let tokenized_array=[];
    for(let word of words){
        let arr=[];
      for(let i=0;i<word.length;i++){
arr.push(word[i]);
      }
tokenized_array.push(arr);
    }
return tokenized_array;
}


let tokenized_text_array=tokenize("lower lowest");



const countPairs=(words)=> {

    let pairCounts = new Map();

    for (let word of words) {

        for (let i = 0; i < word.length - 1; i++) {

            let pair = word[i] + word[i + 1];

            pairCounts.set(
                pair,
                (pairCounts.get(pair) || 0) + 1
            );

        }
    }

    return pairCounts;
}

let pairCounts=countPairs(tokenized_text_array);



function getMostFrequentPair(pairCounts) {

    let bestPair = null;
    let maxCount = 0;

    for (let [pair, count] of pairCounts) {

        if (count > maxCount) {

            maxCount = count;
            bestPair = pair;

        }
    }

    return bestPair;
}


let pair=getMostFrequentPair(pairCounts);


function mergePair(tokenized_text_array,pair){

    let merged_pair_array=[];

    for(let word of tokenized_text_array){
        for(let i=0;i<pair.length;i++){
            let ans='';
            if(word[i]==[pair[i]]){
             ans=ans+word[i];
            }
             
            
        }
    }


}





