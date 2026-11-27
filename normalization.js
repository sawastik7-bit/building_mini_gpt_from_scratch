
function normalize(input_matrix,gamma_matrix,beta_matrix) {

    let mean_matrix = [];

    // STEP 1: Calculate mean of each row
    for (let i = 0; i < input_matrix.length; i++) {

        let mean = 0;

        for (let j = 0; j < input_matrix[i].length; j++) {
            mean += input_matrix[i][j];
        }

        mean_matrix.push(mean / input_matrix[i].length);
    }

    // STEP 2: Calculate variance of each row
    let variance_matrix = [];

    for (let i = 0; i < input_matrix.length; i++) {

        let variance = 0;

        for (let j = 0; j < input_matrix[i].length; j++) {

            let difference = input_matrix[i][j] - mean_matrix[i];

            variance += difference * difference;
        }

        variance_matrix.push(variance / input_matrix[i].length);
    }

    // STEP 3: Normalize each value
    let normalized_matrix = [];

    const epsilon = 1e-5;

    for (let i = 0; i < input_matrix.length; i++) {

        let normalized_row = [];

        let standard_deviation = Math.sqrt(
            variance_matrix[i] + epsilon
        );

        for (let j = 0; j < input_matrix[i].length; j++) {

            let normalized_value =
                (input_matrix[i][j] - mean_matrix[i]) /
                standard_deviation;
                 
                const adjusted_value =
    gamma_matrix[j] * normalized_value + beta_matrix[j];

normalized_row.push(adjusted_value);
            normalized_row.push(normalized_value);
        }

        normalized_matrix.push(normalized_row);
    }

    return normalized_matrix;
}
