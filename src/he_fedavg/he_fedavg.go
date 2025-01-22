package main

import (
	"encoding/json"
	"fmt"
	"math"
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

func main() {
	// ====================================================================
	// FLClient: Load plaintext weights from JSON (after training in python)
	// ====================================================================
	fmt.Println("--- FLClient: Load plaintext weights from JSON (after training in python) ---")
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
	fmt.Println("Model 1")
	print2DLayerDimensions(weights.FC1)
	print2DLayerDimensions(weights.FC2)
	fmt.Println("Model 2")
	print2DLayerDimensions(weights2.FC1)
	print2DLayerDimensions(weights2.FC2)
	fmt.Println("Model 3")
	print2DLayerDimensions(weights3.FC1)
	print2DLayerDimensions(weights3.FC2)

	// Flatten the weights
	weights.FC1_flatten = flatten2D(weights.FC1)
	weights.FC2_flatten = flatten2D(weights.FC2)
	weights2.FC1_flatten = flatten2D(weights2.FC1)
	weights2.FC2_flatten = flatten2D(weights2.FC2)
	weights3.FC1_flatten = flatten2D(weights3.FC1)
	weights3.FC2_flatten = flatten2D(weights3.FC2)
	fmt.Printf("weights.FC1_flatten len %d \n", len(weights.FC1_flatten))
	fmt.Printf("weights.FC2_flatten len %d \n", len(weights.FC2_flatten))

	// =================================
	// Debugging: Plaintext Averaging
	// =================================
	fmt.Println("--- Debugging: Plaintext Averaging ---")
	wantAvgFC1 := make([]float64, len(weights.FC1_flatten))
	wantAvgFC2 := make([]float64, len(weights.FC2_flatten))
	for i := 0; i < len(weights.FC1_flatten); i++ {
		wantAvgFC1[i] = (weights.FC1_flatten[i] + weights2.FC1_flatten[i] + weights3.FC1_flatten[i])
		wantAvgFC1[i] *= 1.0 / 3.0
	}
	for i := 0; i < len(weights.FC2_flatten); i++ {
		wantAvgFC2[i] = (weights.FC2_flatten[i] + weights2.FC2_flatten[i] + weights3.FC2_flatten[i])
		wantAvgFC2[i] *= 1.0 / 3.0
	}

	// =================================
	// FLClient: Instantiating the ckks.Parameters
	// =================================
	fmt.Println("--- FLClient: Instantiating the CKKS Parametersckks.Parameters ---")
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

	// ==========================================
	// FLClient / Trusted Keys Dealer: Keys Generation
	// ==========================================
	fmt.Println("--- Trusted Keys Dealer: CKKS Keys Generation ---")
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// ecd := ckks.NewEncoder(params)
	ecd2 := ckks.NewEncoder(ckks.Parameters(params))

	// ====================================================
	// FLClient: Encrypting the weights homomorphically
	// ====================================================
	fmt.Println("--- FLClient: Encrypting the weights homomorphically ---")
	weights.FC1_encrypted = encryptFlattened(weights.FC1_flatten, Slots, params, ecd2, pk)
	weights.FC2_encrypted = encryptFlattened(weights.FC2_flatten, Slots, params, ecd2, pk)
	weights2.FC1_encrypted = encryptFlattened(weights2.FC1_flatten, Slots, params, ecd2, pk)
	weights2.FC2_encrypted = encryptFlattened(weights2.FC2_flatten, Slots, params, ecd2, pk)
	weights3.FC1_encrypted = encryptFlattened(weights3.FC1_flatten, Slots, params, ecd2, pk)
	weights3.FC2_encrypted = encryptFlattened(weights3.FC2_flatten, Slots, params, ecd2, pk)

	// loop through the weights and print out length and type
	for i := 0; i < len(weights.FC1_encrypted); i++ {
		fmt.Printf("weights.FC1_encrypted[%d] type: %T\n", i, weights.FC1_encrypted[i])
		fmt.Println("metadata: ", weights.FC1_encrypted[i].MetaData)
	}

	// ====================================
	// FLAggregator: Encrypted Averaging
	// ====================================
	fmt.Println("--- FLAggregator: Encrypted Averaging ---")
	eval := ckks.NewEvaluator(params, evk)
	scalar := 1.0 / 3.0

	var encryptedAvgFC1 []*rlwe.Ciphertext
	for i := 0; i < len(weights.FC1_encrypted); i++ {
		temp := weights.FC1_encrypted[i]
		temp, err = eval.AddNew(temp, weights2.FC1_encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.AddNew(temp, weights3.FC1_encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.MulRelinNew(temp, scalar)
		if err != nil {
			panic(err)
		}
		encryptedAvgFC1 = append(encryptedAvgFC1, temp)
	}
	fmt.Println("encryptedAvgFC1FC1 len: ", len(encryptedAvgFC1))

	var encryptedAvgFC2 []*rlwe.Ciphertext
	for i := 0; i < len(weights.FC2_encrypted); i++ {
		temp := weights.FC2_encrypted[i]
		temp, err = eval.AddNew(temp, weights2.FC2_encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.AddNew(temp, weights3.FC2_encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.MulRelinNew(temp, scalar)
		if err != nil {
			panic(err)
		}
		encryptedAvgFC2 = append(encryptedAvgFC2, temp)
	}
	fmt.Println("encryptedAvgFC2 len: ", len(encryptedAvgFC2))

	// ===========================
	// Debugging: Decrypt and Decode
	// ===========================
	fmt.Println("--- Debugging: Decrypt and Decode ---")
	dec := rlwe.NewDecryptor(params, sk)
	var decryptedAvg []float64
	for i := 0; i < len(encryptedAvgFC1); i++ {
		decrypted, err := decryptDecode(dec, ecd2, encryptedAvgFC1[i], params)
		if err != nil {
			panic(err)
		}
		for j := 0; j < len(decrypted); j++ {
			decryptedAvg = append(decryptedAvg, decrypted[j])
		}
	}

	// Calculate the error
	fmt.Printf("wantAvgFC1 len: %d\n", len(wantAvgFC1))
	fmt.Printf("plainSum len: %d\n", len(decryptedAvg))
	// trim sum to have the same length as wantAvgFC1
	decryptedAvg = decryptedAvg[:len(wantAvgFC1)]
	// print the first 10 elements
	fmt.Println("wantAvgFC1: ", wantAvgFC1[:10])
	fmt.Println("decryptedAvg: ", decryptedAvg[:10])

	error := calculateError(decryptedAvg, wantAvgFC1)
	fmt.Printf("Comparing encrypted and plaintext calculations, error = : %f\n", error)

	// ==================================
	// FLAggreagtor: Saving encrypted weights to binary
	// ==================================
	fmt.Println("--- FLAggreagtor: Saving encrypted weights to binary ---")
	for i := 0; i < len(encryptedAvgFC1); i++ {
		bytes, err := encryptedAvgFC1[i].MarshalBinary()
		if err != nil {
			panic(err)
		}
		filename := fmt.Sprintf(weightDir+"avgEncryptedFC1_part%d.bin", i)
		err = os.WriteFile(filename, bytes, 0644)
		if err != nil {
			panic(err)
		}
	}

	bytes, err := encryptedAvgFC2[0].MarshalBinary()
	if err != nil {
		panic(err)
	}
	filename := weightDir + "avgEncryptedFC2.bin"
	err = os.WriteFile(filename, bytes, 0644)
	if err != nil {
		panic(err)
	}

	// ====================================
	// FLClient: Loading encrypted avg weights from binary
	// ====================================
	fmt.Println("--- FLClient: Loading encrypted avg weights from binary ---")
	var loadedEncryptedAvgFC1 []*rlwe.Ciphertext
	var loadedEncryptedAvgFC2 *rlwe.Ciphertext
	for i := 0; i < len(encryptedAvgFC1); i++ {
		filename := fmt.Sprintf(weightDir+"avgEncryptedFC1_part%d.bin", i)
		bytes, err := os.ReadFile(filename)
		if err != nil {
			panic(err)
		}
		// Create a new ciphertext instance
		ct := rlwe.NewCiphertext(params, 1, params.MaxLevel())
		// Unmarshal the binary data into the ciphertext
		err = ct.UnmarshalBinary(bytes)
		if err != nil {
			panic(err)
		}
		loadedEncryptedAvgFC1 = append(loadedEncryptedAvgFC1, ct)
	}

	loadedEncryptedAvgFC2 = rlwe.NewCiphertext(params, 1, params.MaxLevel())
	err = loadedEncryptedAvgFC2.UnmarshalBinary(bytes)
	if err != nil {
		panic(err)
	}

	// =======================================
	// FLClient: Decrypting the loaded weights
	// =======================================
	fmt.Println("--- FLClient: Decrypting the loaded weights ---")
	var decryptedAvgFC1 []float64
	var decryptedAvgFC2 []float64
	for i := 0; i < len(encryptedAvgFC1); i++ {
		decrypted, err := decryptDecode(dec, ecd2, loadedEncryptedAvgFC1[i], params)
		if err != nil {
			panic(err)
		}
		for j := 0; j < len(decrypted); j++ {
			decryptedAvgFC1 = append(decryptedAvgFC1, decrypted[j])
		}
	}
	decryptedAvgFC1 = decryptedAvgFC1[:len(wantAvgFC1)]
	error = calculateError(decryptedAvgFC1, wantAvgFC1)
	fmt.Printf("Comparing decrypted loaded FC1 and plaintext calculations, error = : %f\n", error)

	decrypted, err := decryptDecode(dec, ecd2, loadedEncryptedAvgFC2, params)
	if err != nil {
		panic(err)
	}
	for j := 0; j < len(decrypted); j++ {
		decryptedAvgFC2 = append(decryptedAvgFC2, decrypted[j])
	}
	decryptedAvgFC2 = decryptedAvgFC2[:len(wantAvgFC2)]
	error = calculateError(decryptedAvgFC2, wantAvgFC2)
	fmt.Printf("Comparing decrypted loaded FC2 and plaintext calculations, error = : %f\n", error)

	// ======================================================
	// FLClient: Saving decrypted loaded weights into json
	// ======================================================
	fmt.Println("--- FLClient: Saving decrypted loaded weights into json ---")
	saveToJSON(decryptedAvgFC1, weightDir+"avgDecryptedFC1.json")
	saveToJSON(decryptedAvgFC2, weightDir+"avgDecryptedFC2.json")
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
	// fmt.Printf("numCiphertexts: %d\n", numCiphertexts)
	result := make([]*rlwe.Ciphertext, numCiphertexts)
	for i := 0; i < numCiphertexts; i++ {
		plaintextStart := i * numSlots
		plaintextEnd := (i + 1) * numSlots
		if plaintextEnd > len(plaintext) {
			plaintextEnd = len(plaintext)
		}
		plaintextVec := plaintext[plaintextStart:plaintextEnd]
		// fmt.Printf("plaintextVec number %d len: %d\n", i, len(plaintextVec))
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

func decryptDecode(
	dec *rlwe.Decryptor,
	ecd *ckks.Encoder,
	ct *rlwe.Ciphertext,
	params ckks.Parameters,
) ([]float64, error) {
	dec_pt := dec.DecryptNew(ct)
	// Decodes the plaintext
	have := make([]float64, params.MaxSlots())
	err := ecd.Decode(dec_pt, have)
	if err != nil {
		return nil, err
	}
	return have, nil
}

func calculateError(have []float64, want []float64) float64 {
	var sum float64
	for i := 0; i < len(have); i++ {
		sum += math.Abs(have[i] - want[i])
	}
	return sum
}

func saveToJSON(data []float64, filename string) {
	// Create a file
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Create an encoder and write the data
	encoder := json.NewEncoder(file)
	err = encoder.Encode(data)
	if err != nil {
		panic(err)
	}
}
