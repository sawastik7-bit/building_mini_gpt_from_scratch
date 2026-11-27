
export function residualConnection(originalMatrix, projectedMatrix) {

    if (
        originalMatrix.length !== projectedMatrix.length ||
        originalMatrix[0].length !== projectedMatrix[0].length
    ) {
        throw new Error("Matrix addition is not possible");
    }

    let resulted_residual_matrix = [];

    for (let i = 0; i < originalMatrix.length; i++) {

        resulted_residual_matrix[i] = [];

        for (let j = 0; j < originalMatrix[0].length; j++) {

            resulted_residual_matrix[i][j] =
                originalMatrix[i][j] + projectedMatrix[i][j];

        }
    }

    return resulted_residual_matrix;
}
