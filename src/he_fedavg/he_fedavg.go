package main

import (
	"encoding/json"
	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/utils"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	var err error
	root := FLRubato.FindRootPath()
	logger := utils.NewLogger(utils.DEBUG)
	nMaxElmPrint := 4 // the maximum number of elements we want to be printed when printing a vector
	logger.PrintHeader("FLClient: Load plaintext weights from JSON")
	weightDir := filepath.Join(root, configs.MNIST)

	weights := utils.NewModelWeights()
	err = weights.LoadWeights(weightDir + "/mnist_weights_exclude_137.json")
	utils.HandleError(err)

	weights2 := utils.NewModelWeights()
	err = weights2.LoadWeights(weightDir + "/mnist_weights_exclude_258.json")
	utils.HandleError(err)

	weights3 := utils.NewModelWeights()
	err = weights3.LoadWeights(weightDir + "/mnist_weights_exclude_469.json")
	utils.HandleError(err)

	// Print the weights' dimensions
	logger.PrintMessage("Model 1")
	weights.Print2DLayerDimension(logger)

	logger.PrintMessage("Model 2")
	weights2.Print2DLayerDimension(logger)

	logger.PrintMessage("Model 3")
	weights3.Print2DLayerDimension(logger)

	logger.PrintFormatted("weights.FC1_flatten len: %d", len(weights.FC1Flatten))
	logger.PrintFormatted("weights.FC2_flatten len: %d", len(weights.FC2Flatten))

	// =================================
	// Debugging: Plaintext Averaging
	// =================================
	logger.PrintHeader("Debugging: Plaintext Averaging")
	wantAvgFC1 := make([]float64, len(weights.FC1Flatten))
	wantAvgFC2 := make([]float64, len(weights.FC2Flatten))
	for i := 0; i < len(weights.FC1Flatten); i++ {
		wantAvgFC1[i] = weights.FC1Flatten[i] + weights2.FC1Flatten[i] + weights3.FC1Flatten[i]
		wantAvgFC1[i] *= 1.0 / 3.0
	}
	for i := 0; i < len(weights.FC2Flatten); i++ {
		wantAvgFC2[i] = weights.FC2Flatten[i] + weights2.FC2Flatten[i] + weights3.FC2Flatten[i]
		wantAvgFC2[i] *= 1.0 / 3.0
	}

	// =================================
	// FLClient: Instantiating the ckks.Parameters
	// =================================
	logger.PrintHeader("FLClient: Instantiating the CKKS Parameters")
	var params ckks.Parameters
	if params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,                                    // A ring degree of 2^{16}
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45}, // An initial prime of 55 bits and 7 primes of 45 bits
		LogP:            []int{61},                             // The log2 size of the key-switching prime
		LogDefaultScale: 45,                                    // The default log2 of the scaling factor
	}); err != nil {
		panic(err)
	}
	prec := params.EncodingPrecision() // we will need this value later

	logger.PrintFormatted("Precision: %d", prec)
	LogSlots := params.LogMaxSlots()
	Slots := 1 << LogSlots
	logger.PrintFormatted("Number of slots: %d", Slots)

	// ==========================================
	// FLClient / Trusted Keys Dealer: Keys Generation
	// ==========================================
	logger.PrintHeader("Trusted Keys Dealer: CKKS Keys Generation")
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// ecd := ckks.NewEncoder(params)
	ecd2 := ckks.NewEncoder(params)

	// ====================================================
	// FLClient: Encrypting the weights homomorphically
	// ====================================================
	logger.PrintHeader("FLClient: Encrypting the weights homomorphically")
	weights.FC1Encrypted = encryptFlattened(weights.FC1Flatten, Slots, params, ecd2, pk)
	weights.FC2Encrypted = encryptFlattened(weights.FC2Flatten, Slots, params, ecd2, pk)
	weights2.FC1Encrypted = encryptFlattened(weights2.FC1Flatten, Slots, params, ecd2, pk)
	weights2.FC2Encrypted = encryptFlattened(weights2.FC2Flatten, Slots, params, ecd2, pk)
	weights3.FC1Encrypted = encryptFlattened(weights3.FC1Flatten, Slots, params, ecd2, pk)
	weights3.FC2Encrypted = encryptFlattened(weights3.FC2Flatten, Slots, params, ecd2, pk)

	// loop through the weights and print out length and type
	for i := 0; i < len(weights.FC1Encrypted); i++ {
		logger.PrintFormatted("weights.FC1_encrypted[%d] type: %T", i, weights.FC1Encrypted[i])
		logger.PrintMessages("metadata: ", weights.FC1Encrypted[i].MetaData)
		// for getting more clear information you can use this
		//metaData, _ := weights.FC1Encrypted[i].MetaData.MarshalJSON()
		//logger.PrintMessages("metadata: ", string(metaData))
	}

	// ====================================
	// FLAggregator: Encrypted Averaging
	// ====================================
	logger.PrintHeader("FLAggregator: Encrypted Averaging")
	eval := ckks.NewEvaluator(params, evk)
	scalar := 1.0 / 3.0

	var encryptedAvgFC1 []*rlwe.Ciphertext
	for i := 0; i < len(weights.FC1Encrypted); i++ {
		temp := weights.FC1Encrypted[i]
		temp, err = eval.AddNew(temp, weights2.FC1Encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.AddNew(temp, weights3.FC1Encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.MulRelinNew(temp, scalar)
		if err != nil {
			panic(err)
		}
		encryptedAvgFC1 = append(encryptedAvgFC1, temp)
	}
	logger.PrintMessages("encryptedAvgFC1FC1 len: ", len(encryptedAvgFC1))

	var encryptedAvgFC2 []*rlwe.Ciphertext
	for i := 0; i < len(weights.FC2Encrypted); i++ {
		temp := weights.FC2Encrypted[i]
		temp, err = eval.AddNew(temp, weights2.FC2Encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.AddNew(temp, weights3.FC2Encrypted[i])
		if err != nil {
			panic(err)
		}
		temp, err = eval.MulRelinNew(temp, scalar)
		if err != nil {
			panic(err)
		}
		encryptedAvgFC2 = append(encryptedAvgFC2, temp)
	}
	logger.PrintMessages("encryptedAvgFC2 len: ", len(encryptedAvgFC2))

	// ===========================
	// Debugging: Decrypt and Decode
	// ===========================
	logger.PrintHeader("Debugging: Decrypt and Decode")
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
	logger.PrintFormatted("wantAvgFC1 len: %d", len(wantAvgFC1))
	logger.PrintFormatted("plainSum len: %d", len(decryptedAvg))
	// trim sum to have the same length as wantAvgFC1
	decryptedAvg = decryptedAvg[:len(wantAvgFC1)]
	logger.PrintMessages("wantAvgFC1: ", wantAvgFC1[:nMaxElmPrint])
	logger.PrintMessages("decryptedAvg: ", decryptedAvg[:nMaxElmPrint])

	diff := calculateError(decryptedAvg, wantAvgFC1)
	logger.PrintFormatted("Comparing encrypted and plaintext calculations, error = : %f", diff)

	// ==================================
	// FLAggreagtor: Saving encrypted weights to binary
	// ==================================
	logger.PrintHeader("FLAggreagtor: Saving encrypted weights to binary")
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
	logger.PrintHeader("FLClient: Loading encrypted avg weights from binary")
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
	logger.PrintHeader("FLClient: Decrypting the loaded weights")
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
	diff = calculateError(decryptedAvgFC1, wantAvgFC1)
	logger.PrintFormatted("Comparing decrypted loaded FC1 and plaintext calculations, error = : %f", diff)

	decrypted, err := decryptDecode(dec, ecd2, loadedEncryptedAvgFC2, params)
	if err != nil {
		panic(err)
	}
	for j := 0; j < len(decrypted); j++ {
		decryptedAvgFC2 = append(decryptedAvgFC2, decrypted[j])
	}
	decryptedAvgFC2 = decryptedAvgFC2[:len(wantAvgFC2)]
	diff = calculateError(decryptedAvgFC2, wantAvgFC2)
	logger.PrintFormatted("Comparing decrypted loaded FC2 and plaintext calculations, error = : %f", diff)

	// ======================================================
	// FLClient: Saving decrypted loaded weights into json
	// ======================================================
	logger.PrintHeader("FLClient: Saving decrypted loaded weights into json")
	saveToJSON(decryptedAvgFC1, weightDir+"avgDecryptedFC1.json")
	saveToJSON(decryptedAvgFC2, weightDir+"avgDecryptedFC2.json")
}

func encryptFlattened(plaintext []float64, numSlots int, params ckks.Parameters, encoder *ckks.Encoder, pk *rlwe.PublicKey) []*rlwe.Ciphertext {
	numCiphertexts := len(plaintext) / numSlots
	if len(plaintext)%numSlots != 0 {
		numCiphertexts += 1
	}
	// logger.PrintFormatted("numCiphertexts: %d", numCiphertexts)
	result := make([]*rlwe.Ciphertext, numCiphertexts)
	for i := 0; i < numCiphertexts; i++ {
		plaintextStart := i * numSlots
		plaintextEnd := (i + 1) * numSlots
		if plaintextEnd > len(plaintext) {
			plaintextEnd = len(plaintext)
		}
		plaintextVec := plaintext[plaintextStart:plaintextEnd]
		// logger.PrintFormatted("plaintextVec number %d len: %d", i, len(plaintextVec))
		ciphertext := encryptVec(plaintextVec, params, encoder, pk)
		result[i] = ciphertext
	}
	return result
}

// HE encrypts a plaintext vector of float64 using the provided parameters and public key
func encryptVec(plaintext []float64, params ckks.Parameters, encoder *ckks.Encoder, pk *rlwe.PublicKey) *rlwe.Ciphertext {
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

func decryptDecode(dec *rlwe.Decryptor, ecd *ckks.Encoder, ct *rlwe.Ciphertext, params ckks.Parameters) ([]float64, error) {
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
