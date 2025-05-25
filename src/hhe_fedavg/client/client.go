package client

import (
	"crypto/rand"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/utils"
)

type FLClient struct {
	ClientID      string
	Nonces        [][]byte
	Counter       []byte
	KeyStream     [][]uint64
	SymmCipher    []*RtF.PlaintextRingT
	PlaintextData [][]float64 // for debug
}

func RunFLClient(
	logger utils.Logger,
	rootPath string,
	params *keys_dealer.RubatoParams,
	hheComponents *keys_dealer.HHEComponents,
	weightPath string,
	clientID string,
) *FLClient {
	logger.PrintHeader(fmt.Sprintf("--- Client %s ---", clientID))
	logger.PrintMessage("[Client - Initialization]: Load plaintext weights from JSON")

	keysDir := filepath.Join(rootPath, configs.Keys)

	modelWeights := utils.OpenModelWeights(logger, rootPath, weightPath)
	modelWeights.Print2DLayerDimension(logger)

	logger.PrintMessage("[Client] Preparing the data")
	outputSize := 2 // 1 for FC1 and 1 for FC2
	var data [][]float64 = PreparingData(logger, outputSize, params.Params, modelWeights)
	logger.PrintFormatted("Data.shape = [%d][%d]", len(data), len(data[0]))

	logger.PrintMessage("[Client - Offline] Generating the nonces")
	nonces := make([][]byte, params.Params.N())
	for i := range params.Params.N() {
		nonces[i] = make([]byte, 64)
		rand.Read(nonces[i])
	}
	logger.PrintFormatted("Nonces diminsion: [%d][%d]", len(nonces), len(nonces[0]))

	logger.PrintMessage("[Client - Offline] Generating counter")
	counter := make([]byte, 64)
	rand.Read(counter)
	logger.PrintFormatted("Counter diminsion: [%d]", len(counter))

	logger.PrintMessage("[Client - Offline] Loading the symmetric key")
	symKeyPath := filepath.Join(keysDir, configs.SymmetricKey)
	symKey := keys_dealer.LoadSymmKey(symKeyPath, params.Blocksize)

	logger.PrintMessage("[Client - Offline] Generating the keystream z")
	t := time.Now()
	keystream := make([][]uint64, params.Params.N())
	for i := range params.Params.N() {
		keystream[i] = RtF.PlainRubato(
			params.Blocksize,
			params.NumRound,
			nonces[i],
			counter,
			symKey,
			params.Params.PlainModulus(),
			params.Sigma)
	}
	logger.PrintRunningTime("Time to generate the keystream", t)

	t = time.Now()
	logger.PrintMessage("[Client - Online] Encrypting the plaintext data using the symmetric key stream")
	plainCKKSRingTs := EncryptData(logger, params, hheComponents.CkksEncoder, data, keystream)
	logger.PrintRunningTime("Time to encrypting the plaintext data using the symmetric key stream", t)

	// Save the symmetric encrypted data
	t = time.Now()
	logger.PrintMessage("[Client - Online] Saving the symmetric encrypted data")
	ciphertextDir := filepath.Join(rootPath, configs.SymmetricEncryptedWeights)
	os.MkdirAll(ciphertextDir, 0755)
	SavePlaintextRingTArray(logger, plainCKKSRingTs, ciphertextDir, clientID)
	logger.PrintRunningTime("Time to save the symmetric encrypted data", t)

	return &FLClient{
		ClientID:      clientID,
		Nonces:        nonces,
		Counter:       counter,
		KeyStream:     keystream,
		SymmCipher:    plainCKKSRingTs,
		PlaintextData: data,
	}
}

func PreparingData(logger utils.Logger, outputSize int, params *RtF.Parameters, mw utils.ModelWeights) [][]float64 {
	data := make([][]float64, outputSize)

	logger.PrintFormatted("The data structure is [%d][%d] ([outputSize][params.N()])", outputSize, params.N())
	logger.PrintFormatted("We have the flatten weights as [%d] (for FC1) and [%d] (for FC2)",
		len(mw.FC1Flatten), len(mw.FC2Flatten))

	cnt := 0

	// start with FC1
	cipherPerFC1 := int(math.Ceil(float64(len(mw.FC1Flatten)) / float64(params.N())))
	paddingLenFC1 := params.N() - (len(mw.FC1Flatten) / cipherPerFC1)
	fc1Space := params.N() - paddingLenFC1
	logger.PrintFormatted("Number of ciphers required to store FC1 = len(FC1Flatten) / params.N(): %d", cipherPerFC1)
	logger.PrintFormatted("FC1 Space: %d", fc1Space)
	logger.PrintFormatted("Padding length for each cipher required to store FC1: %d", paddingLenFC1)
	if cipherPerFC1 > 0 {
		for i := range cipherPerFC1 {
			data[cnt] = make([]float64, params.N())
			for j := range params.N() {
				if j < fc1Space {
					data[cnt][j] = mw.FC1Flatten[(i*fc1Space)+j]
				} else {
					data[cnt][j] = float64(0) // for padding
				}
			}
			cnt++ // moving to the next plaintext slot (row)
		}
	} else {
		log.Fatalln("Something is wrong with the input data length!")
	}

	logger.PrintMessages("FC1 data space and the padding: ", data[cnt-1][fc1Space-4:fc1Space+4])

	// then FC2
	cipherPerFC2 := int(math.Ceil(float64(len(mw.FC2Flatten)) / float64(params.N())))
	paddingLenFC2 := params.N() - (len(mw.FC2Flatten) / cipherPerFC2)
	fc2Space := params.N() - paddingLenFC2
	logger.PrintFormatted("Number of ciphers required to store FC2: %d", cipherPerFC2)
	logger.PrintFormatted("FC2 Space: %d", fc2Space)
	logger.PrintFormatted("Padding length required to store FC2: %d", paddingLenFC2)
	if cipherPerFC2 > 0 {
		for i := range cipherPerFC2 {
			data[cnt] = make([]float64, params.N())
			for j := range params.N() {
				if j < fc2Space {
					data[cnt][j] = mw.FC2Flatten[(i*fc2Space)+j]
				} else {
					data[cnt][j] = float64(0) // for padding
				}
			}
			cnt++ // moving to the next plaintext slot (row)
		}
	} else {
		log.Fatalln("Something is wrong with the input data length!")
	}
	logger.PrintMessages("FC2 data space and the padding: ", data[cnt-1][fc2Space-4:fc2Space+4])

	return data
}

func EncryptData(
	logger utils.Logger,
	params *keys_dealer.RubatoParams,
	ckksEncoder RtF.CKKSEncoder,
	data [][]float64,
	keystream [][]uint64) []*RtF.PlaintextRingT {
	logger.PrintMessage("[Client - Online] Move data to the plaintext's coefficients")
	coefficients := make([][]float64, params.OutputSize)
	for s := range params.OutputSize {
		coefficients[s] = make([]float64, params.Params.N())
	}

	// Copy data to coefficients with bit-reversal
	for s := range params.OutputSize {
		for i := range params.Params.N() {
			j := utils.BitReverse64(uint64(i), uint64(params.Params.LogN()-1))
			if i < params.Params.N()/2 {
				coefficients[s][j] = data[s][i]
				coefficients[s][j+uint64(params.Params.N()/2)] = data[s][i+params.Params.N()/2]
			}
		}
	}

	logger.PrintMessage("[Client - Online] Encrypting the plaintext data using the symmetric key stream")
	plainCKKSRingTs := make([]*RtF.PlaintextRingT, params.OutputSize)
	for s := range params.OutputSize {
		logger.PrintMessage("Scale up the plaintext message -> m̃")
		plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coefficients[s], params.MessageScaling) // scales up the plaintext message
		poly := plainCKKSRingTs[s].Value()[0]
		logger.PrintMessage("Modulo q addition between the keystream z and the scaled message -> c_{ctr}")
		for i := range params.Params.N() {
			j := utils.BitReverse64(uint64(i), uint64(params.Params.LogN()))
			poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.Params.PlainModulus() // modulo q addition between the keystream to the scaled message
		}
	}
	logger.PrintFormatted("Symmetric encrypted data: %+T, len = %d", plainCKKSRingTs, len(plainCKKSRingTs))

	return plainCKKSRingTs
}

// SavePlaintextRingTArray saves an array of plaintexts to individual files in a directory
func SavePlaintextRingTArray(logger utils.Logger, plaintexts []*RtF.PlaintextRingT, dirPath string, clientID string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	// Save length file
	lengthPath := filepath.Join(dirPath, fmt.Sprintf("%s_length.txt", clientID))
	if err := os.WriteFile(lengthPath, []byte(strconv.Itoa(len(plaintexts))), 0644); err != nil {
		return fmt.Errorf("failed to write length file: %v", err)
	}

	// Save each plaintext
	for i, pt := range plaintexts {
		if pt == nil {
			return fmt.Errorf("plaintext at index %d is nil", i)
		}

		fileName := fmt.Sprintf("%s_pt_%d.bin", clientID, i)
		filePath := filepath.Join(dirPath, fileName)

		if err := utils.Serialize(pt, filePath); err != nil {
			return fmt.Errorf("failed to save plaintext %d: %v", i, err)
		}
	}

	logger.PrintFormatted("Symmetric encrypted data saved to %s", dirPath)

	return nil
}

// LoadPlaintextRingTArray loads an array of plaintexts from a directory
func LoadPlaintextRingTArray(dirPath string, clientID string, params *RtF.Parameters) []*RtF.PlaintextRingT {
	// Read length file
	lengthPath := filepath.Join(dirPath, fmt.Sprintf("%s_length.txt", clientID))
	lengthBytes, err := os.ReadFile(lengthPath)
	if err != nil {
		panic(fmt.Errorf("failed to read length file: %v", err))
	}

	length, err := strconv.Atoi(string(lengthBytes))
	if err != nil {
		panic(fmt.Errorf("failed to parse length: %v", err))
	}

	// Create array to hold plaintexts
	plaintexts := make([]*RtF.PlaintextRingT, length)

	// Load each plaintext
	for i := range length {
		fileName := fmt.Sprintf("%s_pt_%d.bin", clientID, i)
		filePath := filepath.Join(dirPath, fileName)

		// Create new plaintext object
		plaintexts[i] = RtF.NewPlaintextRingT(params)

		// Deserialize into it
		if err := utils.Deserialize(plaintexts[i], filePath); err != nil {
			panic(fmt.Errorf("failed to load plaintext %d: %v", i, err))
		}
	}

	return plaintexts
}
