package main

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/utils"
)

//================= How Rubato Works: =================//
/*
	1. Generating the HE context keys
	2. Generating HalfBoot Keys
	3. Data Generation (or Collection)
		- Rubato, depending on the parameter selection, has a fixed block size as BS={16, 36, 64}
		|-> with corresponding plaintext modulus logP: {26, 25, 25} and LogN = 16, logSlots = 15,
		|-> with a scaling factor logSF = {45}. Therefore, we can make a matrix as following
		|-> data = [BS-4][N]float64.
	4. Generation of the SymKey, Nonce, and Counter
		- The SymKey will be as K = [BS]uint64
		- The Nonce will be as nonces = [N][8]byte
		- The Counter will be as counter = [8]byte
	5. Symmetric keystream generation
		- Using plainRubato(BS, numRound, nonces[i], counter, symkey, pMod, sigma) we can generate
		|-> a keystream as ks = [N][BS-4]uint64, (Each element of keystream has 4 columns less than
		|-> the actual block size, since we drop 4 elements because of the noise addition)
	6. Symmetric Data Encryption
		- To encrypt the data using the symmetric keystream, we first move data to the plaintext's
		|-> coefficients and then use CKKS Encode function to have data as a CKKS polynomial ring.
		|-> Finally, one can add the keystream to the polynomial coefficients and do the modular reduction.
	7. Scaling up the data
		- In this step, we Scale Up the encrypted data, PlaintextRingT (R_p) into a Plaintext (R_q),
		|-> so we get a plaintext in RingQ with level=0
	8. Encrypting the SymKey
		- In this step, we will encrypt the SymKey homomorphically using MFVRubato as kCT
	9. Offline computation
 		- This step includes the generation of homomorphic keystream using kCT and then switching
		|-> the modulus down to stcModDown, so in the end we will have the homomorphic keystream
		|-> ciphertext with the same modulus as encrypted data
	10. Online computation
		- This step includes the homomorphic decryption of the symmetrically encrypted data (aka
		|-> transciphering) to have homomorphically encrypted data (CKKS ciphertext) using HalfBoot
		|-> function. One can then use the result ciphertext to do the rest of the HE application
		|->	evaluations. Keep in mind that one can also store this ciphertext to avoid the repeat of
		|-> the heavy computation for transciphering.
*/
//================= The End xD =================//
func main() {
	logger := utils.NewLogger(utils.DEBUG)
	root := FLRubato.FindRootPath()

	logger.PrintHeader("[Client - Initialization]: Load plaintext weights from JSON (after training in python)")
	mws := make([]utils.ModelWeights, 3)
	mws[0] = utils.OpenModelWeights(logger, root, "mnist_weights_exclude_137.json")
	mws[1] = utils.OpenModelWeights(logger, root, "mnist_weights_exclude_258.json")
	mws[2] = utils.OpenModelWeights(logger, root, "mnist_weights_exclude_469.json")

	for _, mw := range mws {
		mw.Print2DLayerDimension(logger)
	}

	Rubato(logger, root, RtF.RUBATO128L, mws, true) // Always use fullCoffs = true

}

// Rubato is the one
func Rubato(logger utils.Logger, root string, paramIndex int, mws []utils.ModelWeights, fullCoffs bool) {
	keysDir := filepath.Join(root, configs.Keys)
	ciphersDir := filepath.Join(root, configs.Ciphertexts)

	// MFV instances
	var halfBootstrapper *RtF.HalfBootstrapper
	var fvEncoder RtF.MFVEncoder
	var ckksEncoder RtF.CKKSEncoder
	var ckksDecryptor RtF.CKKSDecryptor
	var fvEncryptor RtF.MFVEncryptor
	var fvEvaluator RtF.MFVEvaluator
	var plainCKKSRingTs []*RtF.PlaintextRingT
	var plaintexts []*RtF.Plaintext

	// Rubato parameter
	logger.PrintHeader("[Client - Initialization] HHE parameters")
	blockSize := RtF.RubatoParams[paramIndex].Blocksize
	// outputSize := blockSize - 4
	outputSize := 3
	numRound := RtF.RubatoParams[paramIndex].NumRound
	plainModulus := RtF.RubatoParams[paramIndex].PlainModulus
	sigma := RtF.RubatoParams[paramIndex].Sigma
	logger.PrintFormatted("blockSize = %d", blockSize)
	logger.PrintFormatted("outputSize = %d", outputSize)
	logger.PrintFormatted("numRound = %d", numRound)
	logger.PrintFormatted("plainModulus = %d", plainModulus)
	logger.PrintFormatted("sigma = %f", sigma)

	// RtF Rubato parameters, for full-coefficients only (128bit security)
	halfBsParams := RtF.RtFRubatoParams[0]
	params, err := halfBsParams.Params()
	if err != nil {
		panic(err)
	}
	params.SetPlainModulus(plainModulus)
	messageScaling := float64(params.PlainModulus()) / halfBsParams.MessageRatio

	rubatoModDown := RtF.RubatoModDownParams[paramIndex].CipherModDown
	stcModDown := RtF.RubatoModDownParams[paramIndex].StCModDown

	// fullCoffs denotes whether full coefficients are used for data encoding
	// NOTE: we should always use fullCoffs=true for now
	if fullCoffs {
		params.SetLogFVSlots(params.LogN())
		logger.PrintFormatted("params.N() = %d", params.N())
	} else {
		params.SetLogFVSlots(params.LogSlots())
	}

	logger.PrintHeader("[Client - Initialization] HHE keys generation")
	HHEKeysGen(logger, keysDir, params, halfBsParams, fullCoffs)

	// reading the already generated keys from a previous step, it will save time and memory :)
	fvEncoder, ckksEncoder, fvEncryptor, ckksDecryptor, halfBootstrapper, fvEvaluator = InitHHEScheme(
		logger,
		keysDir,
		params,
		halfBsParams,
	)

	// Rubato instance
	rubato := RtF.NewMFVRubato(paramIndex, params, fvEncoder, fvEncryptor, fvEvaluator, rubatoModDown[0])

	var data [][]float64
	logger.PrintHeader("[Client] Preparing the data")
	data = preparingData(logger, outputSize, params, mws[0])

	// Allocating the coefficients
	coefficients := make([][]float64, outputSize)
	for s := 0; s < outputSize; s++ {
		coefficients[s] = make([]float64, params.N())
	}

	logger.PrintHeader("[Client - Initialization] Symmetric Key Generation and Encryption")
	key, kCt, err := SymmetricKeyGen(logger, keysDir, blockSize, params, rubato)
	if err != nil {
		log.Fatalf("Failed to generate symmetric key: %v", err)
	}

	var nonces [][]byte
	var counter []byte
	var keystream [][]uint64

	logger.PrintHeader("[Client - Offline] Generating the nonces and counter")
	nonces = make([][]byte, params.N())
	for i := 0; i < params.N(); i++ {
		nonces[i] = make([]byte, 64)
		rand.Read(nonces[i])
	}
	logger.PrintFormatted("Nonces diminsion: [%d][%d]", len(nonces), len(nonces[0]))

	counter = make([]byte, 64)
	rand.Read(counter)
	logger.PrintFormatted("Counter diminsion: [%d]", len(counter))

	logger.PrintHeader("[Client - Offline] Generating the keystream z")
	keystream = make([][]uint64, params.N())
	for i := 0; i < params.N(); i++ {
		keystream[i] = RtF.PlainRubato(blockSize, numRound, nonces[i], counter, key, params.PlainModulus(), sigma)
	}

	for s := 0; s < outputSize; s++ {
		for i := 0; i < params.N()/2; i++ {
			j := utils.BitReverse64(uint64(i), uint64(params.LogN()-1))
			coefficients[s][j] = data[s][i]
			coefficients[s][j+uint64(params.N()/2)] = data[s][i+params.N()/2]
		}
	}

	logger.PrintHeader("[Client - Online] Encrypting the plaintext data using the symmetric key stream")
	plainCKKSRingTs = make([]*RtF.PlaintextRingT, outputSize)
	for s := 0; s < outputSize; s++ {
		plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coefficients[s], messageScaling) // scales up the plaintext message
		poly := plainCKKSRingTs[s].Value()[0]
		for i := 0; i < params.N(); i++ {
			j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
			poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus() // modulo q addition between the keystream to the scaled message
		}
	}

	var fvKeyStreams []*RtF.Ciphertext
	//fvKeyStreams = rubato.Crypt(nonces, counter, kCt, rubatoModDown)
	logger.PrintHeader("[Server - Offline] Evaluates the keystreams (Eval^{FV} to produce V")
	t := time.Now()
	fvKeyStreams = rubato.CryptNoModSwitch(nonces, counter, kCt) // Compute ciphertexts without modulus switching
	logger.PrintRunningTime("Time to evaluate the keystreams (Eval^{FV}) to produce V", t)

	logger.PrintHeader("[Server - Offline] Performs linear transformation SlotToCoeffs^{FV} to produce Z")
	t = time.Now()
	for i := 0; i < outputSize; i++ {
		fvKeyStreams[i] = fvEvaluator.SlotsToCoeffs(fvKeyStreams[i], stcModDown)
		fvEvaluator.ModSwitchMany(fvKeyStreams[i], fvKeyStreams[i], fvKeyStreams[i].Level())
	}
	logger.PrintRunningTime("Time to perform linear transformation SlotToCoeffs^{FV} to produce Z", t)

	logger.PrintHeader("[Server - Online] Scale up the symmetric ciphertext (Scale{FV}) into FV-ciphretext space (produce C)")
	t = time.Now()
	plaintexts = make([]*RtF.Plaintext, outputSize)
	for s := 0; s < outputSize; s++ {
		plaintexts[s] = RtF.NewPlaintextFVLvl(params, 0)
		fvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}
	logger.PrintRunningTime("Time to scale up the symmetric ciphertext into FV-ciphertext space", t)

	var ctBoot *RtF.Ciphertext
	logger.PrintHeader("[Server - Online] Transciphering the symmetric ciphertext into CKKS ciphertext (produce M)")
	for s := 0; s < outputSize; s++ {
		// Encrypt and mod switch to the lowest level
		ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
		ciphertext.Value()[0] = plaintexts[s].Value()[0].CopyNew()
		logger.PrintMessage("Subtracting the homomorphically evaluated keystream Z from the symmetric ciphertext C (produce X)")
		fvEvaluator.Sub(ciphertext, fvKeyStreams[s], ciphertext)
		fvEvaluator.TransformToNTT(ciphertext, ciphertext)
		ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(params.Qi()[0]) / float64(params.PlainModulus()) * messageScaling))))
		// Half-Bootstrap the ciphertext (homomorphic evaluation of ModRaise -> SubSum -> CtS -> EvalMod)
		// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
		// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
		// The difference from the bootstrapping is that the last StC is missing.
		// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
		// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
		if fullCoffs {
			logger.PrintMessage("Halfboot X and outputs a CKKS-ciphertext M containing the CKKS encrypted messages in its slots")
			t = time.Now()
			ctBoot, _ = halfBootstrapper.HalfBoot(ciphertext, false)
			logger.PrintRunningTime("HalfBoot", t)
		} else {
			ctBoot, _ = halfBootstrapper.HalfBoot(ciphertext, true)
		}

		valuesWant := make([]complex128, params.Slots())
		for i := 0; i < params.Slots(); i++ {
			valuesWant[i] = complex(data[s][i], 0)
		}

		printString := fmt.Sprintf("Precision of HalfBoot(ciphertext[%d])", s)
		logger.PrintHeader(printString)
		printDebug(logger, params, ctBoot, valuesWant, ckksDecryptor, ckksEncoder)

		// save the CKKS ciphertext in a file for further computation
		SaveCipher(logger, s, ciphersDir, ctBoot)
	}
}

func preparingData(logger utils.Logger, outputSize int, params *RtF.Parameters, mw utils.ModelWeights) [][]float64 {
	data := make([][]float64, outputSize)

	logger.PrintFormatted("The data structure is [%d][%d] ([outputSize][params.N()])", outputSize, params.N())
	logger.PrintFormatted("We have the flatten weights as [%d] (for FC1) and [%d] (for FC2)",
		len(mw.FC1Flatten), len(mw.FC2Flatten))

	cnt := 0 // will use this counter for locating
	// basically for each flattened FCx we will take as much full ciphertext space as it needs,
	// for example, here, for 128L security, the output size is 60, so we have 60* ciphers each with
	// 65536 elements. The FC1 has 100352 elements; therefore, we need 2 full ciphertext spaces to
	// put it there. Of course, there will be some free space, which we use padding and 0 value.
	// The data will be like:
	// [0][FC1:Padding]
	// [1][FC1:Padding]
	// [2][FC2:Padding]
	// [0][FC1:Padding]
	// [1][FC1:Padding]
	// [2][FC2:Padding]
	//	...
	// [60][Padding]

	// start with FC1
	cipherPerFC1 := int(math.Ceil(float64(len(mw.FC1Flatten)) / float64(params.N()))) // MNIST: 2
	paddingLenFC1 := params.N() - (len(mw.FC1Flatten) / cipherPerFC1)                 // MNIST: 15360
	fc1Space := params.N() - paddingLenFC1                                            // MNIST: 50176
	logger.PrintFormatted("Number of ciphers required to store FC1: %d", cipherPerFC1)
	logger.PrintFormatted("FC1 Space: %d", fc1Space)
	logger.PrintFormatted("Padding length for each cipher required to store FC1: %d", paddingLenFC1)
	if cipherPerFC1 > 0 {
		for i := 0; i < cipherPerFC1; i++ {
			data[cnt] = make([]float64, params.N())
			for j := 0; j < params.N(); j++ {
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
		for i := 0; i < cipherPerFC2; i++ {
			data[cnt] = make([]float64, params.N())
			for j := 0; j < params.N(); j++ {
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

	// filling the rest with padding (this is not efficient at all) -> the solution will be changing the parameters
	// for s := cnt; s < outputSize; s++ {
	// 	data[s] = make([]float64, params.N())
	// 	for i := 0; i < params.N(); i++ {
	// 		data[s][i] = float64(0)
	// 	}
	// }
	return data
}

// SaveCipher save a ciphertext in the provided path
func SaveCipher(
	logger utils.Logger,
	index int,
	ciphersDir string,
	ciphertext *RtF.Ciphertext) {
	logger.PrintHeader("Save the CKKS ciphertext in a file for further computation")
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(index) + configs.CtFormat
	err = utils.Serialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
}

// LoadCipher save a ciphertext in the provided path
func LoadCipher(
	logger utils.Logger,
	fileIndex int,
	ciphersDir string,
	params *RtF.Parameters) *RtF.Ciphertext {
	logger.PrintHeader("Load the CKKS ciphertext from the file")
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(fileIndex) + configs.CtFormat
	ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
	err = utils.Deserialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	return ciphertext
}

// Generates and saves cryptographic keys for Homomorphic Hybrid Encryption (HHE).
func HHEKeysGen(
	logger utils.Logger,
	keysDir string,
	params *RtF.Parameters,
	hbtParams *RtF.HalfBootParameters,
	fullCoffs bool) {

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
		logger.PrintMessage("Some key files already exist, skipping keys generation")
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
	if !fullCoffs {
		rotations = append(rotations, params.Slots()/2)
	}

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

// SymmetricKeyGen generates a symmetric key and its corresponding FV ciphertext
// If the key and ciphertext already exist in storage, it loads and returns them.
func SymmetricKeyGen(
	logger utils.Logger,
	keysDir string,
	blockSize int,
	params *RtF.Parameters,
	rubato RtF.MFVRubato) (key []uint64, kCt []*RtF.Ciphertext, err error) {

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
		key, err := LoadSymmKey(symKeyPath, blockSize)
		if err != nil {
			fmt.Printf("Failed to load key: %v\n", err)
			return nil, nil, fmt.Errorf("failed to load symmetric key: %v", err)
		}

		t := time.Now()
		// Load ciphertext array kCt
		kCt, err := LoadCiphertextArray(symCipherDir, params)
		if err != nil {
			fmt.Printf("Failed to load ciphertext array: %v\n", err)
			return nil, nil, fmt.Errorf("failed to load FV ciphertext symmetric key: %v", err)
		}
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

// InitHHEScheme loads the homomorphic hybrid encryption keys from storage and initializes
// the complete cryptographic scheme including encoders, encryptors, decryptors, evaluators,
// and the half-bootstrapping components. It returns all necessary components for HHE operations.
func InitHHEScheme(
	logger utils.Logger,
	keysDir string,
	params *RtF.Parameters,
	hbtpParams *RtF.HalfBootParameters) (encoder RtF.MFVEncoder,
	ckksEncoder RtF.CKKSEncoder,
	encryptor RtF.MFVEncryptor,
	decryptor RtF.CKKSDecryptor,
	halfBootstrapper *RtF.HalfBootstrapper,
	fvEvaluator RtF.MFVEvaluator) {
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
	if halfBootstrapper, err = RtF.NewHalfBootstrapper(params, hbtpParams, hbtpKey); err != nil {
		panic(err)
	}
	logger.PrintMemUsage("New HalfBootstrapper")

	encoder = RtF.NewMFVEncoder(params)
	ckksEncoder = RtF.NewCKKSEncoder(params)
	encryptor = RtF.NewMFVEncryptorFromPk(params, pk)
	decryptor = RtF.NewCKKSDecryptor(params, sk)

	ptDiagMat := encoder.GenSlotToCoeffMatFV(2)
	logger.PrintMemUsage("PtDiagMatrix Generation")
	fvEvaluator = RtF.NewMFVEvaluator(params, RtF.EvaluationKey{Rlk: rlKeys, Rtks: rotKeys}, ptDiagMat)
	logger.PrintRunningTime("Total time to load the keys: ", t)

	return encoder, ckksEncoder, encryptor, decryptor, halfBootstrapper, fvEvaluator
}

func printDebug(
	logger utils.Logger,
	params *RtF.Parameters,
	ciphertext *RtF.Ciphertext,
	valuesWant []complex128,
	decryptor RtF.CKKSDecryptor,
	encoder RtF.CKKSEncoder) {
	if utils.DEBUG {

		valuesTest := encoder.DecodeComplex(decryptor.DecryptNew(ciphertext), params.LogSlots())
		logSlots := params.LogSlots()
		sigma := params.Sigma()

		logger.PrintFormatted("Level: %d (logQ = %d)", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
		logger.PrintFormatted("Scale: 2^%f", math.Log2(ciphertext.Scale()))
		logger.PrintFormatted("ValuesTest{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(valuesTest), valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
		logger.PrintFormatted("ValuesWant{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(valuesWant), valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

		precisionStats := RtF.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, logSlots, sigma)
		fmt.Println(precisionStats.String())

	}

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

// LoadKey loads a sequential key array from a file
func LoadSymmKey(filePath string, blockSize int) ([]uint64, error) {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Create the key slice
	key := make([]uint64, blockSize)

	// Read each uint64 value
	for i := 0; i < blockSize; i++ {
		if err := binary.Read(file, binary.LittleEndian, &key[i]); err != nil {
			return nil, fmt.Errorf("failed to read key at position %d: %v", i, err)
		}
	}

	return key, nil
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

// LoadCiphertextArray loads an array of ciphertexts from a directory
// params is needed to create new ciphertext objects
func LoadCiphertextArray(dirPath string, params *RtF.Parameters) ([]*RtF.Ciphertext, error) {
	// Read length file
	lengthPath := filepath.Join(dirPath, "length.txt")
	lengthBytes, err := os.ReadFile(lengthPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read length file: %v", err)
	}

	length, err := strconv.Atoi(string(lengthBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to parse length: %v", err)
	}

	// Create array to hold ciphertexts
	ciphertexts := make([]*RtF.Ciphertext, length)

	// Load each ciphertext
	for i := 0; i < length; i++ {
		fileName := fmt.Sprintf("ct_%d.bin", i)
		filePath := filepath.Join(dirPath, fileName)

		// Create new ciphertext object
		ct := RtF.NewCiphertextFVLvl(params, 1, 0)

		// Deserialize into it
		if err := utils.Deserialize(ct, filePath); err != nil {
			return nil, fmt.Errorf("failed to load ciphertext %d: %v", i, err)
		}

		ciphertexts[i] = ct
	}

	return ciphertexts, nil
}
