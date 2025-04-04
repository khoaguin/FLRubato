/// The keys dealer is responsible for generating the Rubato parameters, HHE keys,
/// symmetric keys and their corresponding FV ciphertexts

package keys_dealer

import (
	"encoding/binary"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/utils"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type RubatoParams struct {
	Blocksize      int
	OutputSize     int
	NumRound       int
	PlainModulus   uint64
	Sigma          float64
	MessageScaling float64
	RubatoModDown  []int
	StcModDown     []int
	HalfBsParams   *RtF.HalfBootParameters
	Params         *RtF.Parameters
}

type HHEComponents struct {
	FvEncoder        RtF.MFVEncoder
	CkksEncoder      RtF.CKKSEncoder
	FvEncryptor      RtF.MFVEncryptor
	CkksDecryptor    RtF.CKKSDecryptor
	FvEvaluator      RtF.MFVEvaluator
	HalfBootstrapper *RtF.HalfBootstrapper
	CkksEvaluator    RtF.CKKSEvaluator
}

func RunKeysDealer(
	logger utils.Logger,
	rootPath string,
	paramIndex int) (
	rubatoParams *RubatoParams,
	hheComponents *HHEComponents,
	rubato RtF.MFVRubato,
) {
	logger.PrintHeader("--- Keys Dealer ---")
	logger.PrintHeader("[Keys Dealer] Preparing Common things for all FL Clients")
	logger.PrintFormatted("Root Path: %s", rootPath)
	logger.PrintFormatted("Parameter Index: %d", paramIndex)

	// Initialize Rubato parameters
	rubatoParams = InitRubatoParams(logger, paramIndex)

	// Initialize HHE components
	keysDir := filepath.Join(rootPath, configs.Keys)

	// Create keysDir if it doesn't exist
	if err := os.MkdirAll(keysDir, 0755); err != nil {
		utils.HandleError(fmt.Errorf("failed to create keys directory: %v", err))
	}
	logger.PrintFormatted("Keys directory: %s", keysDir)

	HHEKeysGen(logger, keysDir, rubatoParams.Params, rubatoParams.HalfBsParams)

	// reading the already generated keys from a previous step, it will save time and memory :)
	hheComponents = InitHHEScheme(
		logger, keysDir, rubatoParams.Params, rubatoParams.HalfBsParams,
	)

	rubato = RtF.NewMFVRubato(
		paramIndex,
		rubatoParams.Params,
		hheComponents.FvEncoder,
		hheComponents.FvEncryptor,
		hheComponents.FvEvaluator,
		rubatoParams.RubatoModDown[0],
	)

	var err error
	symKey, symKeyFVCiphertext, err := SymmetricKeyGen(
		logger, keysDir, rubatoParams.Blocksize, rubatoParams.Params, rubato,
	)
	if err != nil {
		utils.HandleError(err)
	}
	logger.PrintFormatted("Symmetric Key: %+v", symKey)
	logger.PrintFormatted("FV encrypted symmetric key: %+v", symKeyFVCiphertext)

	return rubatoParams, hheComponents, rubato
}

func InitRubatoParams(logger utils.Logger, paramIndex int) *RubatoParams {
	logger.PrintHeader("[Keys Dealer] Rubato parameters")
	blockSize := RtF.RubatoParams[paramIndex].Blocksize
	outputSize := 3 // originally: outputSize := blockSize - 4
	numRound := RtF.RubatoParams[paramIndex].NumRound
	plainModulus := RtF.RubatoParams[paramIndex].PlainModulus
	sigma := RtF.RubatoParams[paramIndex].Sigma

	// RtF Rubato parameters, for full-coefficients only (128bit security)
	halfBsParams := RtF.RtFRubatoParams[0]
	params, err := halfBsParams.Params()
	if err != nil {
		utils.HandleError(err)
	}
	params.SetPlainModulus(plainModulus)
	messageScaling := float64(params.PlainModulus()) / halfBsParams.MessageRatio

	rubatoModDown := RtF.RubatoModDownParams[paramIndex].CipherModDown
	stcModDown := RtF.RubatoModDownParams[paramIndex].StCModDown

	params.SetLogFVSlots(params.LogN())

	logger.PrintFormatted("blockSize = %d", blockSize)
	logger.PrintFormatted("outputSize = %d", outputSize)
	logger.PrintFormatted("numRound = %d", numRound)
	logger.PrintFormatted("plainModulus = %d", plainModulus)
	logger.PrintFormatted("sigma = %f", sigma)
	logger.PrintFormatted("params.N() = %d", params.N())
	logger.PrintFormatted("params.Slots() = %d", params.Slots())

	return &RubatoParams{
		Blocksize:      blockSize,
		OutputSize:     outputSize,
		NumRound:       numRound,
		PlainModulus:   plainModulus,
		Sigma:          sigma,
		MessageScaling: messageScaling,
		RubatoModDown:  rubatoModDown,
		StcModDown:     stcModDown,
		HalfBsParams:   halfBsParams,
		Params:         params,
	}
}

// Generates and saves cryptographic keys for Homomorphic Hybrid Encryption (HHE).
func HHEKeysGen(
	logger utils.Logger,
	keysDir string,
	params *RtF.Parameters,
	hbtParams *RtF.HalfBootParameters,
) {
	logger.PrintHeader("[Keys Dealer] HHE keys generation")

	var err error

	// Check if all required files exist
	secretKeyPath := filepath.Join(keysDir, configs.SecretKey)
	publicKeyPath := filepath.Join(keysDir, configs.PublicKey)
	rotationKeyPath := filepath.Join(keysDir, configs.RotationKeys)
	relinKeysPath := filepath.Join(keysDir, configs.RelinearizationKeys)

	fileExists := func(path string) bool {
		_, err := os.Stat(path)
		return !os.IsNotExist(err)
	}

	if fileExists(secretKeyPath) || fileExists(publicKeyPath) ||
		fileExists(rotationKeyPath) || fileExists(relinKeysPath) {
		logger.PrintFormatted("Some key files already exist in %s, skipping keys generation", keysDir)
		return
	}

	kgen := RtF.NewKeyGenerator(params)

	logger.PrintMemUsage("Secret and Public Keys Generation")
	sk, pk := kgen.GenKeyPairSparse(hbtParams.H)
	err = utils.Serialize(sk, filepath.Join(keysDir, configs.SecretKey))
	utils.HandleError(err)
	err = utils.Serialize(pk, filepath.Join(keysDir, configs.PublicKey))
	utils.HandleError(err)

	fvEncoder := RtF.NewMFVEncoder(params)

	// Generating half-bootstrapping keys
	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(params.LogSlots(), hbtParams)

	t := time.Now()
	ptDiagMats := fvEncoder.GenSlotToCoeffMatFV(2) // radix = 2
	logger.PrintMemUsage("StC Matrix Generation")
	logger.PrintRunningTime("StC Matrix Generation", t)

	//We cannot serialize the ptDiagMats, it doesn't support serialization
	//err = utils.Serialize(ptDiagMats, filepath.Join(keysDir, configs.StCDiagMatrix))
	//utils.HandleError(err)

	t = time.Now()
	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(ptDiagMats)
	logger.PrintMemUsage("Rotation Indices Generation")
	logger.PrintRunningTime("Rotation Indices Generation", t)
	rotations := append(rotationsHalfBoot, rotationsStC...)

	t = time.Now()
	rotKeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	logger.PrintMemUsage("Rotation Keys Generation")
	logger.PrintRunningTime("Rotation Keys Generation", t)
	err = utils.Serialize(rotKeys, filepath.Join(keysDir, configs.RotationKeys))
	utils.HandleError(err)

	t = time.Now()
	rlk := kgen.GenRelinearizationKey(sk)
	logger.PrintMemUsage("Relinearization Keys Generation")
	logger.PrintRunningTime("Relinearization Keys Generation", t)
	err = utils.Serialize(rlk, filepath.Join(keysDir, configs.RelinearizationKeys))
	utils.HandleError(err)
}

// InitHHEScheme loads the homomorphic hybrid encryption keys from storage and initializes
// the complete cryptographic scheme including encoders, encryptors, decryptors, evaluators,
// and the half-bootstrapping components. It returns all necessary components for HHE operations.
func InitHHEScheme(
	logger utils.Logger,
	keysDir string,
	params *RtF.Parameters,
	hbtpParams *RtF.HalfBootParameters) *HHEComponents {
	logger.PrintHeader("[Keys Dealer] Initializing HHE Scheme")

	logger.PrintHeader("Reading the keys and public parameters from storage and setup the scheme")
	t := time.Now()
	var err error

	sk := new(RtF.SecretKey)
	if err = utils.Deserialize(sk, filepath.Join(keysDir, configs.SecretKey)); err != nil {
		utils.HandleError(err)
	}
	logger.PrintMemUsage("Reading sk")

	pk := new(RtF.PublicKey)
	if err = utils.Deserialize(pk, filepath.Join(keysDir, configs.PublicKey)); err != nil {
		utils.HandleError(err)
	}
	logger.PrintMemUsage("Reading pk")

	rotKeys := new(RtF.RotationKeySet)
	if err = utils.Deserialize(rotKeys, filepath.Join(keysDir, configs.RotationKeys)); err != nil {
		utils.HandleError(err)
	}
	logger.PrintMemUsage("Reading rotKeys")

	rlKeys := new(RtF.RelinearizationKey)
	if err = utils.Deserialize(rlKeys, filepath.Join(keysDir, configs.RelinearizationKeys)); err != nil {
		utils.HandleError(err)
	}
	logger.PrintMemUsage("Reading rlKeys")

	hbtpKey := RtF.BootstrappingKey{Rlk: rlKeys, Rtks: rotKeys}
	halfBootstrapper, err := RtF.NewHalfBootstrapper(params, hbtpParams, hbtpKey)
	if err != nil {
		panic(err)
	}

	logger.PrintMemUsage("New HalfBootstrapper")

	fvEncoder := RtF.NewMFVEncoder(params)
	ckksEncoder := RtF.NewCKKSEncoder(params)
	fvEncryptor := RtF.NewMFVEncryptorFromPk(params, pk)
	ckksDecryptor := RtF.NewCKKSDecryptor(params, sk)

	ptDiagMat := fvEncoder.GenSlotToCoeffMatFV(2)
	logger.PrintMemUsage("PtDiagMatrix Generation")
	fvEvaluator := RtF.NewMFVEvaluator(params, RtF.EvaluationKey{Rlk: rlKeys, Rtks: rotKeys}, ptDiagMat)
	logger.PrintRunningTime("Total time to load the keys: ", t)

	ckksEvaluator := RtF.NewCKKSEvaluator(params, RtF.EvaluationKey{Rlk: rlKeys, Rtks: rotKeys})

	return &HHEComponents{
		FvEncoder:        fvEncoder,
		CkksEncoder:      ckksEncoder,
		FvEncryptor:      fvEncryptor,
		CkksDecryptor:    ckksDecryptor,
		HalfBootstrapper: halfBootstrapper,
		FvEvaluator:      fvEvaluator,
		CkksEvaluator:    ckksEvaluator,
	}
}

// SymmetricKeyGen generates a symmetric key and its corresponding FV ciphertext
// If the key and ciphertext already exist in storage, it loads and returns them.
func SymmetricKeyGen(
	logger utils.Logger,
	keysDir string,
	blockSize int,
	params *RtF.Parameters,
	rubato RtF.MFVRubato) (key []uint64, kCt []*RtF.Ciphertext, err error) {
	logger.PrintHeader("[Keys Dealer] Generating / Loading Symmetric Keys")

	symKeyPath := filepath.Join(keysDir, configs.SymmetricKey)
	symCipherDir := filepath.Join(keysDir, configs.SymmetricKeyCipherDir)

	fileExists := func(path string) bool {
		_, err := os.Stat(path)
		return !os.IsNotExist(err)
	}

	dirExists := func(path string) bool {
		info, err := os.Stat(path)
		return !os.IsNotExist(err) && info.IsDir()
	}

	if fileExists(symKeyPath) && dirExists(symCipherDir) {
		logger.PrintMessage("Loading existing symmetric key and ciphertext")

		// Load symmetric key
		key = LoadSymmKey(symKeyPath, blockSize)

		t := time.Now()
		// Load ciphertext array kCt
		kCt = LoadCiphertextArray(symCipherDir, params)
		logger.PrintRunningTime("Time to load the symmetric key FV ciphertext", t)

		return key, kCt, nil
	}

	// Generate new symmetric key
	t := time.Now()
	key = make([]uint64, blockSize)
	for i := 0; i < blockSize; i++ {
		key[i] = uint64(i + 1) // Use (1, ..., 16) for testing
	}
	logger.PrintRunningTime("Symmetric Key Generation", t)

	// Save symmetric key
	if err := SaveSymmKey(key, symKeyPath); err != nil {
		fmt.Printf("Failed to save key: %v\n", err)
		return nil, nil, fmt.Errorf("failed to save symmetric key: %v", err)
	}
	logger.PrintFormatted("Symmetric key saved to %s", symKeyPath)

	// Compute FV Ciphertext of the symmetric key
	logger.PrintMessage("Compute FV Ciphertext of the Symmetric Key")
	t = time.Now()
	kCt = rubato.EncKey(key)
	logger.PrintRunningTime("Time to compute FV Ciphertext of the Symmetric Key: ", t)

	// Save ciphertext array kCt
	if err := SaveCiphertextArray(kCt, symCipherDir); err != nil {
		fmt.Printf("Failed to save ciphertext array: %v\n", err)
		return nil, nil, fmt.Errorf("failed to save FV ciphertext symmetric key: %v", err)
	}
	logger.PrintFormatted("FV Ciphertext of the Symmetric key saved to %s", symCipherDir)

	return key, kCt, nil
}

// LoadCiphertextArray loads an array of ciphertexts from a directory
// params is needed to create new ciphertext objects
func LoadCiphertextArray(dirPath string, params *RtF.Parameters) []*RtF.Ciphertext {
	// Read length file
	lengthPath := filepath.Join(dirPath, "length.txt")
	lengthBytes, err := os.ReadFile(lengthPath)
	if err != nil {
		panic(fmt.Errorf("failed to read length file: %v", err))
	}

	length, err := strconv.Atoi(string(lengthBytes))
	if err != nil {
		panic(fmt.Errorf("failed to parse length: %v", err))
	}

	// Create array to hold ciphertexts
	ciphertexts := make([]*RtF.Ciphertext, length)

	// Load each ciphertext
	for i := range length {
		fileName := fmt.Sprintf("ct_%d.bin", i)
		filePath := filepath.Join(dirPath, fileName)

		// Create new ciphertext object
		ct := RtF.NewCiphertextFVLvl(params, 1, 0)

		// Deserialize into it
		if err := utils.Deserialize(ct, filePath); err != nil {
			panic(fmt.Errorf("failed to load ciphertext %d: %v", i, err))
		}

		ciphertexts[i] = ct
	}

	return ciphertexts
}

// LoadKey loads a sequential key array from a file
func LoadSymmKey(filePath string, blockSize int) []uint64 {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		panic(fmt.Errorf("failed to open file: %v", err))
	}
	defer file.Close()

	// Create the key slice
	key := make([]uint64, blockSize)

	// Read each uint64 value
	for i := 0; i < blockSize; i++ {
		if err := binary.Read(file, binary.LittleEndian, &key[i]); err != nil {
			panic(fmt.Errorf("failed to read key at position %d: %v", i, err))
		}
	}

	return key
}

// SaveCiphertextArray saves an array of ciphertexts to individual files in a directory
// Each ciphertext is saved with format "ct_%d.bin" where %d is the index
func SaveCiphertextArray(ciphertexts []*RtF.Ciphertext, dirPath string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	// Save length file
	lengthPath := filepath.Join(dirPath, "length.txt")
	if err := os.WriteFile(lengthPath, []byte(strconv.Itoa(len(ciphertexts))), 0644); err != nil {
		return fmt.Errorf("failed to write length file: %v", err)
	}

	// Save each ciphertext
	for i, ct := range ciphertexts {
		if ct == nil {
			return fmt.Errorf("ciphertext at index %d is nil", i)
		}

		fileName := fmt.Sprintf("ct_%d.bin", i)
		filePath := filepath.Join(dirPath, fileName)

		if err := utils.Serialize(ct, filePath); err != nil {
			return fmt.Errorf("failed to save ciphertext %d: %v", i, err)
		}
	}

	return nil
}

// SaveKey saves a sequential key array to a file
func SaveSymmKey(key []uint64, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	// Create or truncate the file
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write each uint64 in binary format
	for _, val := range key {
		if err := binary.Write(file, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write key: %v", err)
		}
	}

	return nil
}
