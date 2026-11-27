import { residualConnection } from "./residualConnection";
import {initializeWeights, matrixMultiply} from "./selfAttention";
function feedForwardNetwork(input_matrix){


const W1=initializeWeights(4,3);
const b1=new Array(3).fill(0);

const firstLinearOutput=matrixMultiply(input_matrix,W1);

const hidden=addBias(firstLinearOutput,b1);

  const activated = applyGELU(hidden);

  const W2 = initializeWeights(3, 4);
const b2 = new Array(4).fill(0);

const secondLinearOutput = matrixMultiply(activated, W2);
const ffnOutput = addBias(secondLinearOutput, b2);
return ffnOutput;

const finalOutput=residualConnection(input_matrix,ffnOutput);

return finalOutput;
}


function addBias(matrix, bias) {

    if (matrix[0].length !== bias.length) {
        throw new Error("Bias dimension must match matrix columns");
    }

    return matrix.map(row =>
        row.map((value, j) => value + bias[j])
    );
}



function applyGELU(matrix) {
    return matrix.map(row =>
        row.map(value => gelu(value))
    );
}





function gelu(x) {
    return 0.5 * x * (
        1 + Math.tanh(
            Math.sqrt(2 / Math.PI) *
            (x + 0.044715 * Math.pow(x, 3))
        )
    );
}