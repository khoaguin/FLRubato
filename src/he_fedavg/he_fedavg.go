package main

import (
	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/utils"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	NUM_CLIENTS = 3
)

func main() {
	startTime := time.Now()
	RunHEFedAvg()
	endTime := time.Now()
	logger := utils.NewLogger(utils.DEBUG)
	logger.PrintHeader("Time to run HEFedAvg")
	logger.PrintFormatted("Time taken: %f (s)", endTime.Sub(startTime).Seconds())
}

func RunHEFedAvg() {
	root := FLRubato.FindRootPath()
	logger := utils.NewLogger(utils.DEBUG)
	plaintextWeightDir := filepath.Join(root, configs.PlaintextWeights)

	decryptedWeightDir := filepath.Join(root, configs.DecryptedWeights)
	if err := os.MkdirAll(decryptedWeightDir, 0755); err != nil {
		panic(err)
	}

	plainHEEncryptedWeightsDir := filepath.Join(root, configs.HEEncryptedWeights, "plain_he")

	// ---- Keys Dealer ----
	logger.PrintHeader("Keys Dealer")
	ckksParams, _, _, Slots := keysDealerCKKSParams(logger, true)
	sk, pk, evk, ckksEncoder := keysDealerKeysGen(logger, ckksParams, true)

	// ---- Clients ----
	logger.PrintHeader("Clients")
	weights := clientWeights(logger, plaintextWeightDir, true)
	t := time.Now()
	encryptedWeights := clientEncryptWeights(logger, weights, Slots, ckksParams, ckksEncoder, pk, true, true, plainHEEncryptedWeightsDir)
	logger.PrintRunningTime("Time to encrypt the weights homomorphically", t)

	// -- Aggregator Server --
	logger.PrintHeader("Aggregator Server")
	t = time.Now()
	encryptedAvg := aggregatorEncryptedFedAvg(logger, encryptedWeights, ckksParams, evk)
	logger.PrintRunningTime("Time for aggregator server to aggregate the encrypted weights", t)

	// -- Debugging --
	logger.PrintHeader("Testing values")
	plaintextAvgFC1, plaintextAvgFC2 := plaintextAveraging(logger, weights)
	t = time.Now()
	decryptedAvgFC1 := decryptAndDecode(logger, ckksEncoder, encryptedAvg.FC1Encrypted, ckksParams, sk)
	decryptedAvgFC2 := decryptAndDecode(logger, ckksEncoder, encryptedAvg.FC2Encrypted, ckksParams, sk)
	logger.PrintRunningTime("Time to decrypt and decode the encrypted average weights", t)

	decryptedAvgFC1 = decryptedAvgFC1[:len(plaintextAvgFC1)]
	decryptedAvgFC2 = decryptedAvgFC2[:len(plaintextAvgFC2)]

	diff := calculateError(decryptedAvgFC1, plaintextAvgFC1)
	logger.PrintFormatted("Comparing encrypted and plaintext calculations, error = : %f", diff)

	diff = calculateError(decryptedAvgFC2, plaintextAvgFC2)
	logger.PrintFormatted("Comparing encrypted and plaintext calculations, error = : %f", diff)

	// Save the plaintext and decrypted averages to JSON files
	utils.SaveToJSON(logger, decryptedWeightDir, "he_decrypted_avg_fc1.json", decryptedAvgFC1)
	utils.SaveToJSON(logger, decryptedWeightDir, "he_decrypted_avg_fc2.json", decryptedAvgFC2)
}

func keysDealerCKKSParams(
	logger utils.Logger,
	verbose bool,
) (ckks.Parameters, uint, int, int) {
	logger.PrintMessage("[Key Dealer]: CKKS Parameters")
	var ckksParams ckks.Parameters
	var err error
	if ckksParams, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,                                    // A ring degree of 2^{16}
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45}, // An initial prime of 55 bits and 7 primes of 45 bits
		LogP:            []int{61},                             // The log2 size of the key-switching prime
		LogDefaultScale: 45,                                    // The default log2 of the scaling factor
	}); err != nil {
		panic(err)
	}
	encodingPrecision := ckksParams.EncodingPrecision() // we will need this value later
	LogSlots := ckksParams.LogMaxSlots()
	Slots := 1 << LogSlots

	if verbose {
		logger.PrintFormatted("CKKS Parameters: %+v", ckksParams)
		logger.PrintFormatted("Encoding Precision: %d", encodingPrecision)
		logger.PrintFormatted("Log Slots: %d", LogSlots)
		logger.PrintFormatted("Slots: %d", Slots)
	}

	return ckksParams, encodingPrecision, LogSlots, Slots
}

func keysDealerKeysGen(
	logger utils.Logger,
	ckksParams ckks.Parameters,
	verbose bool,
) (*rlwe.SecretKey, *rlwe.PublicKey, rlwe.EvaluationKeySet, *ckks.Encoder) {
	logger.PrintMessage("[Key Dealer]: Keys Generation")
	kgen := rlwe.NewKeyGenerator(ckksParams)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	ecd := ckks.NewEncoder(ckksParams)
	if verbose {
		logger.PrintFormatted("Secret Key Binary Size: %d", sk.BinarySize())
		logger.PrintFormatted("Public Key Binary Size: %d", pk.BinarySize())
		logger.PrintFormatted("Relinearization key type: %T", rlk)
		logger.PrintFormatted("Evaluation Key Set type: %T", evk)
		logger.PrintFormatted("Encoder type: %T", ecd)
	}
	return sk, pk, evk, ecd
}

func clientLoadWeights(
	logger utils.Logger,
	weightDir string,
	weightPath string,
	verbose bool,
) utils.ModelWeights {
	logger.PrintMessage("[Client - Initialization]: Load plaintext weights from JSON")
	var err error
	weights := utils.NewModelWeights()
	err = weights.LoadWeights(weightDir + weightPath)
	utils.HandleError(err)
	if verbose {
		weights.Print2DLayerDimension(logger)
		logger.PrintFormatted("weights.FC1_flatten len: %d", len(weights.FC1Flatten))
		logger.PrintFormatted("weights.FC2_flatten len: %d", len(weights.FC2Flatten))
	}
	return weights
}

func clientWeights(
	logger utils.Logger,
	weightDir string,
	verbose bool,
) []utils.ModelWeights {
	weights1 := clientLoadWeights(logger, weightDir, "/weights_no_137.json", verbose)
	weights2 := clientLoadWeights(logger, weightDir, "/weights_no_258.json", verbose)
	weights3 := clientLoadWeights(logger, weightDir, "/weights_no_469.json", verbose)

	weights := []utils.ModelWeights{weights1, weights2, weights3}
	return weights
}

func clientEncryptWeights(
	logger utils.Logger,
	weights []utils.ModelWeights,
	slots int,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	pk *rlwe.PublicKey,
	verbose bool,
	save bool,
	savedWeightsDir string,
) []utils.ModelWeights {
	logger.PrintMessage("[Client]: Encrypting the weights homomorphically")
	for i := range weights {
		weights[i].FC1Encrypted = encryptFlattened(weights[i].FC1Flatten, slots, ckksParams, ecd, pk)
		weights[i].FC2Encrypted = encryptFlattened(weights[i].FC2Flatten, slots, ckksParams, ecd, pk)
		if verbose {
			logger.PrintFormatted("weights[%d].FC1_encrypted type: %T", i, weights[i].FC1Encrypted)
			logger.PrintFormatted("weights[%d].FC2_encrypted type: %T", i, weights[i].FC2Encrypted)
			// logger.PrintMessages("metadata: ", weights[i].FC1Encrypted[i].MetaData)
		}
		if save {
			clientWeightDir := filepath.Join(savedWeightsDir, fmt.Sprintf("do_%d", i+1)) // do stands for data owner
			SaveEncryptedWeights(logger, weights[i].FC1Encrypted, clientWeightDir, "he_encrypted_fc1")
			SaveEncryptedWeights(logger, weights[i].FC2Encrypted, clientWeightDir, "he_encrypted_fc2")
		}
	}
	return weights
}

func plaintextAveraging(
	logger utils.Logger,
	weights []utils.ModelWeights,
) ([]float64, []float64) {
	logger.PrintMessage("[Debug]: Plaintext Averaging")
	wantAvgFC1 := make([]float64, len(weights[0].FC1Flatten))
	wantAvgFC2 := make([]float64, len(weights[0].FC2Flatten))
	for i := range wantAvgFC1 {
		wantAvgFC1[i] = weights[0].FC1Flatten[i] + weights[1].FC1Flatten[i] + weights[2].FC1Flatten[i]
		wantAvgFC1[i] *= 1.0 / 3.0
	}
	for i := range wantAvgFC2 {
		wantAvgFC2[i] = weights[0].FC2Flatten[i] + weights[1].FC2Flatten[i] + weights[2].FC2Flatten[i]
		wantAvgFC2[i] *= 1.0 / 3.0
	}
	return wantAvgFC1, wantAvgFC2
}

func aggregatorEncryptedFedAvg(
	logger utils.Logger,
	weights []utils.ModelWeights,
	ckksParams ckks.Parameters,
	evk rlwe.EvaluationKeySet,
) utils.ModelWeights {
	logger.PrintMessage("FLAggregator: Encrypted Averaging")
	eval := ckks.NewEvaluator(ckksParams, evk)
	scalar := 1.0 / float64(NUM_CLIENTS)

	// Get the number of ciphertexts per layer (FC1 and FC2)
	numFC1Ciphertexts := len(weights[0].FC1Encrypted)
	numFC2Ciphertexts := len(weights[0].FC2Encrypted)

	// Initialize result arrays
	avgFC1 := make([]*rlwe.Ciphertext, numFC1Ciphertexts)
	avgFC2 := make([]*rlwe.Ciphertext, numFC2Ciphertexts)

	// Process FC1 layer
	for i := range numFC1Ciphertexts {
		// Start with first client's ciphertext
		var err error
		avgFC1[i] = weights[0].FC1Encrypted[i]
		// Add other clients' ciphertexts
		for j := 1; j < NUM_CLIENTS; j++ {
			avgFC1[i], err = eval.AddNew(avgFC1[i], weights[j].FC1Encrypted[i])
			if err != nil {
				panic(err)
			}
		}
		// Multiply by scalar (1/NUM_CLIENTS)
		avgFC1[i], err = eval.MulRelinNew(avgFC1[i], scalar)
		if err != nil {
			panic(err)
		}
	}

	// Process FC2 layer
	for i := range numFC2Ciphertexts {
		// Start with first client's ciphertext
		var err error
		avgFC2[i] = weights[0].FC2Encrypted[i]
		// Add other clients' ciphertexts
		for j := 1; j < NUM_CLIENTS; j++ {
			avgFC2[i], err = eval.AddNew(avgFC2[i], weights[j].FC2Encrypted[i])
			if err != nil {
				panic(err)
			}
		}
		// Multiply by scalar (1/NUM_CLIENTS)
		avgFC2[i], err = eval.MulRelinNew(avgFC2[i], scalar)
		if err != nil {
			panic(err)
		}
	}

	// Create result model weights
	result := utils.NewModelWeights()
	result.FC1Encrypted = avgFC1
	result.FC2Encrypted = avgFC2

	return result
}

func decryptAndDecode(
	logger utils.Logger,
	ecd *ckks.Encoder,
	ciphertext []*rlwe.Ciphertext,
	params ckks.Parameters,
	sk *rlwe.SecretKey,
) []float64 {
	logger.PrintMessage("[Debug]: Decrypt and Decode")
	dec := rlwe.NewDecryptor(params, sk)
	var decryptedAvg []float64
	for i := range ciphertext {
		decrypted, err := decryptDecode(ciphertext[i], dec, ecd, params)
		if err != nil {
			panic(err)
		}
		decryptedAvg = append(decryptedAvg, decrypted...)
	}
	return decryptedAvg
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
	// logger.PrintFormatted("numCiphertexts: %d", numCiphertexts)
	result := make([]*rlwe.Ciphertext, numCiphertexts)
	for i := range numCiphertexts {
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
	ct *rlwe.Ciphertext,
	dec *rlwe.Decryptor,
	ecd *ckks.Encoder,
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
	for i := range have {
		sum += math.Abs(have[i] - want[i])
	}
	return sum
}

// SaveEncryptedWeights saves encrypted weights to binary files
func SaveEncryptedWeights(
	logger utils.Logger,
	weights []*rlwe.Ciphertext,
	outputDir string,
	prefix string,
) {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		panic(err)
	}

	// Save each ciphertext
	for _, ct := range weights {
		fileName := fmt.Sprintf("%s.bin", prefix)
		filePath := filepath.Join(outputDir, fileName)
		if err := utils.Serialize(ct, filePath); err != nil {
			panic(err)
		}
		logger.PrintFormatted("Saved ciphertext to %s", filePath)
	}
}

// LoadEncryptedWeights loads encrypted weights from binary files
func LoadEncryptedWeights(
	logger utils.Logger,
	params ckks.Parameters,
	inputDir string,
	prefix string,
) []*rlwe.Ciphertext {
	var weights []*rlwe.Ciphertext
	i := 0

	// Keep loading ciphertexts until we can't find the next file
	for {
		fileName := fmt.Sprintf("%s_%d.bin", prefix, i)
		filePath := filepath.Join(inputDir, fileName)

		// Check if file exists
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			break
		}

		// Create new ciphertext and load it
		ct := rlwe.NewCiphertext(params, 1, params.MaxLevel())
		if err := utils.Deserialize(ct, filePath); err != nil {
			panic(err)
		}
		weights = append(weights, ct)
		logger.PrintFormatted("Loaded ciphertext from %s", filePath)
		i++
	}

	return weights
}
