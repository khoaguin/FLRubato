package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type ModelWeights struct {
	FC1 [][]float64 `json:"fc1"`
	FC2 [][]float64 `json:"fc2"`
}

func print2DLayerDimensions(layer [][]float64) {
	fmt.Printf("Shape: [%d, %d]\n", len(layer), len(layer[0]))
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

// func inferencePlain(input []float64, weights *ModelWeights) []float64 {
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
	weights, err := loadWeights("./weights/model_weights_exclude_137.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	weights2, err := loadWeights("./weights/model_weights_exclude_258.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	weights3, err := loadWeights("./weights/model_weights_exclude_469.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}

	// Print the weights dimensions
	print2DLayerDimensions(weights.FC1)
	print2DLayerDimensions(weights.FC2)
	print2DLayerDimensions(weights2.FC1)
	print2DLayerDimensions(weights2.FC2)
	print2DLayerDimensions(weights3.FC1)
	print2DLayerDimensions(weights3.FC2)

	// =================================
	// Instantiating the ckks.Parameters
	// =================================
	var params ckks.Parameters
	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            16,                                    // A ring degree of 2^{14}
			LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45}, // An initial prime of 55 bits and 7 primes of 45 bits
			LogP:            []int{61},                             // The log2 size of the key-switching prime
			LogDefaultScale: 45,                                    // The default log2 of the scaling factor
		}); err != nil {
		panic(err)
	}
	prec := params.EncodingPrecision() // we will need this value later

	fmt.Printf("Precision: %d\n", prec)
	LogSlots := params.LogMaxSlots()
	Slots := 1 << LogSlots
	fmt.Printf("Number of slots: %d\n", Slots)

	// ==============
	// Key Generation
	// ==============
	// kgen := rlwe.NewKeyGenerator(params)
	// sk := kgen.GenSecretKeyNew()
	// pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	// rlk := kgen.GenRelinearizationKeyNew(sk)
	// evk := rlwe.NewMemEvaluationKeySet(rlk)

	// ecd2 := ckks.NewEncoder(ckks.Parameters(params))
	// ecd := ckks.NewEncoder(params)

	// enc := rlwe.NewEncryptor(params, pk)

	// for i := 0; i < len(weights.FC1); i++ {
	// 	pt1 := ckks.NewPlaintext(params, params.MaxLevel())
	// 	if err = ecd2.Encode(weights.FC1[i], pt1); err != nil {
	// 		panic(err)
	// 	}
	// 	ct1, err := enc.EncryptNew(pt1)
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	// fmt.Printf("Binary size: %d\n", ct1.BinarySize())
	// }

	// =========================================
	// no way to do matrix encryption in lattigo
	// stick to Rubato params (rtf_params.go)
	// it's better to fill all slots when encrypting
	// For optimized encrypted computations, we do ((128 * 784) / 2**16) = the number of ciphertexts needed
	// then do averaging
}
