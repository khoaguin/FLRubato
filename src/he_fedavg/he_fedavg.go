package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type ModelWeights struct {
	FC1           [][]float64 `json:"fc1"`
	FC2           [][]float64 `json:"fc2"`
	FC1_flatten   []float64
	FC2_flatten   []float64
	FC1_encrypted []*rlwe.Ciphertext
	FC2_encrypted []*rlwe.Ciphertext
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
	weightDir := "../../weights/mnist/"
	weights, err := loadWeights(weightDir + "mnist_weights_exclude_137.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	weights2, err := loadWeights(weightDir + "mnist_weights_exclude_258.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	weights3, err := loadWeights(weightDir + "mnist_weights_exclude_469.json")
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}

	// Print the weights dimensions
	fmt.Println("--- Model Weights ---")
	fmt.Println("Model 1")
	print2DLayerDimensions(weights.FC1)
	print2DLayerDimensions(weights.FC2)
	fmt.Println("Model 2")
	print2DLayerDimensions(weights2.FC1)
	print2DLayerDimensions(weights2.FC2)
	fmt.Println("Model 3")
	print2DLayerDimensions(weights3.FC1)
	print2DLayerDimensions(weights3.FC2)

	// =================================
	// Instantiating the ckks.Parameters
	// =================================
	fmt.Println("--- CKKS Parameters ---")
	var params ckks.Parameters
	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            16,                                    // A ring degree of 2^{16}
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

	// Flatten the weights
	weights.FC1_flatten = flatten2D(weights.FC1)
	weights.FC2_flatten = flatten2D(weights.FC2)
	weights2.FC1_flatten = flatten2D(weights2.FC1)
	weights2.FC2_flatten = flatten2D(weights2.FC2)
	weights3.FC1_flatten = flatten2D(weights3.FC1)
	weights3.FC2_flatten = flatten2D(weights3.FC2)
	fmt.Printf("weights.FC1_flatten len %d \n", len(weights.FC1_flatten))
	fmt.Printf("weights.FC2_flatten len %d \n", len(weights.FC2_flatten))

	// ==============
	// Key Generation
	// ==============
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	// rlk := kgen.GenRelinearizationKeyNew(sk)
	// evk := rlwe.NewMemEvaluationKeySet(rlk)

	// ecd := ckks.NewEncoder(params)
	ecd2 := ckks.NewEncoder(ckks.Parameters(params))

	weights.FC1_encrypted = encryptFlattened(weights.FC1_flatten, Slots, params, ecd2, pk)
	weights.FC2_encrypted = encryptFlattened(weights.FC2_flatten, Slots, params, ecd2, pk)

	// loop through the weights and print out length and type
	for i := 0; i < len(weights.FC1_encrypted); i++ {
		fmt.Printf("weights.FC1_encrypted[%d] type: %T\n", i, weights.FC1_encrypted[i])
		fmt.Println("metadata: ", weights.FC1_encrypted[i].MetaData)
	}
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

// Flatten2D converts a 2D slice into a 1D slice by concatenating all rows (row major packing)
func flatten2D(matrix [][]float64) []float64 {
	// Calculate the total length needed
	totalLen := 0
	for _, row := range matrix {
		totalLen += len(row)
	}

	// Create the flattened slice with the exact capacity needed
	flattened := make([]float64, 0, totalLen)

	// Append all elements
	for _, row := range matrix {
		flattened = append(flattened, row...)
	}

	return flattened
}

func encryptFlattened(
	plaintext []float64,
	numSlots int,
	params ckks.Parameters,
	encoder *ckks.Encoder,
	pk *rlwe.PublicKey,
) []*rlwe.Ciphertext {
	numCiphertexts := len(plaintext) / numSlots
	if len(plaintext)%numSlots != 0 {
		numCiphertexts += 1
	}
	fmt.Printf("numCiphertexts: %d\n", numCiphertexts)
	result := make([]*rlwe.Ciphertext, numCiphertexts)
	for i := 0; i < numCiphertexts; i++ {
		plaintextStart := i * numSlots
		plaintextEnd := (i + 1) * numSlots
		if plaintextEnd > len(plaintext) {
			plaintextEnd = len(plaintext)
		}
		plaintextVec := plaintext[plaintextStart:plaintextEnd]
		fmt.Printf("plaintextVec number %d len: %d\n", i, len(plaintextVec))
		ciphertext := encryptVec(plaintextVec, params, encoder, pk)
		result[i] = ciphertext
	}
	return result
}

// HE encrypts a plaintext vector of float64 using the provided parameters and public key
func encryptVec(
	plaintext []float64,
	params ckks.Parameters,
	encoder *ckks.Encoder,
	pk *rlwe.PublicKey,
) *rlwe.Ciphertext {
	pt1 := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(plaintext, pt1); err != nil {
		panic(err)
	}
	// encrypt
	enc := rlwe.NewEncryptor(params, pk)
	ct1, err := enc.EncryptNew(pt1)
	if err != nil {
		panic(err)
	}

	return ct1
}
