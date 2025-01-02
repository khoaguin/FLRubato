package main

import (
	"encoding/json"
	"fmt"
	"os"
)

type ModelWeights struct {
	FC1        [][]float64 `json:"fc1"`
	FC2        [][]float64 `json:"fc2"`
	InputSize  int         `json:"input_size"`
	HiddenSize int         `json:"hidden_size"`
	OutputSize int         `json:"output_size"`
}

func loadWeights(filename string) (*ModelWeights, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	var weights ModelWeights
	if err := json.Unmarshal(data, &weights); err != nil {
		return nil, fmt.Errorf("error parsing JSON: %v", err)
	}

	return &weights, nil
}

// func inference(input []float64, weights *ModelWeights) []float64 {
// 	if len(input) != weights.InputSize {
// 		panic(fmt.Sprintf("input size mismatch: got %d, expected %d",
// 			len(input), weights.InputSize))
// 	}

// 	output := make([]float64, weights.OutputSize)

// 	// Compute matrix multiplication and add bias
// 	for i := 0; i < weights.OutputSize; i++ {
// 		sum := weights.Bias[i]
// 		for j := 0; j < weights.InputSize; j++ {
// 			sum += input[j] * weights.Weights[i][j]
// 		}
// 		output[i] = sum
// 	}

// 	return output
// }

func main() {
	// Load the weights
	weights, err := loadWeights("weights/model_weights.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}

	// Print the weights
	fmt.Printf("Weights: %+v\n", weights)

	// Create a sample input
	// input := make([]float64, weights.InputSize)
	// for i := range input {
	// 	input[i] = float64(i) / float64(weights.InputSize)
	// }

	// // Run inference
	// output := inference(input, weights)

	// Print results
	// fmt.Println("Input:", input)
	// fmt.Println("Output:", output)
}
