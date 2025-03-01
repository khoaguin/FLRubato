package main

import (
	"crypto/rand"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strconv"
	"time"

	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/utils"

	"flhhe/src/hhe_fedavg/client"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/hhe_fedavg/server"
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
	RunRubato()
	RunFLHHE()
}

func RunRubato() {
	logger := utils.NewLogger(utils.DEBUG)
	rootPath := FLRubato.FindRootPath()

	mws := make([]utils.ModelWeights, 3)
	mws[0] = utils.OpenModelWeights(logger, rootPath, "mnist_weights_exclude_137.json")
	mws[1] = utils.OpenModelWeights(logger, rootPath, "mnist_weights_exclude_258.json")
	mws[2] = utils.OpenModelWeights(logger, rootPath, "mnist_weights_exclude_469.json")

	for _, mw := range mws {
		mw.Print2DLayerDimension(logger)
	}
	Rubato(logger, rootPath, RtF.RUBATO80L, mws)
}

func RunFLHHE() {
	// [WIP] Split this into RunFLClient and RunFLServer
	logger := utils.NewLogger(utils.DEBUG)
	rootPath := FLRubato.FindRootPath()

	paramIndex := RtF.RUBATO128L
	rubatoParams := keys_dealer.InitRubatoParams(logger, paramIndex)
	logger.PrintFormatted("Rubato Parameters: %+v", rubatoParams)
	// logger.PrintFormatted("HHE Components: %+v", hheComponents)
	// logger.PrintFormatted("Rubato Instance: %+v", rubato)

	for round := 0; round < 1; round++ {
		client.RunFLClient(logger, rootPath, rubatoParams.Params, "mnist_weights_exclude_137.json")
		// client.RunFLClient(logger, rootPath, rubatoParams.Params, "mnist_weights_exclude_258.json")
		// client.RunFLClient(logger, rootPath, rubatoParams.Params, "mnist_weights_exclude_469.json")
		server.RunFLServer(logger, rootPath)
	}
}

// Rubato is the one
func Rubato(logger utils.Logger, root string, paramIndex int, mws []utils.ModelWeights) {
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

	params.SetLogFVSlots(params.LogN())
	logger.PrintFormatted("params.N() = %d", params.N())

	keys_dealer.HHEKeysGen(logger, keysDir, params, halfBsParams)

	// reading the already generated keys from a previous step, it will save time and memory :)
	fvEncoder, ckksEncoder, fvEncryptor, ckksDecryptor, halfBootstrapper, fvEvaluator = InitHHEScheme(
		logger,
		keysDir,
		params,
		halfBsParams,
	)

	// Rubato instance
	rubato := RtF.NewMFVRubato(paramIndex, params, fvEncoder, fvEncryptor, fvEvaluator, rubatoModDown[0])

	logger.PrintHeader("[Client] Preparing the data")
	data := client.PreparingData(logger, outputSize, params, mws[0])

	// Allocating the coefficients
	coefficients := make([][]float64, outputSize)
	for s := 0; s < outputSize; s++ {
		coefficients[s] = make([]float64, params.N())
	}

	logger.PrintHeader("[Client - Initialization] Symmetric Key Generation and Encryption")
	key, kCt, err := keys_dealer.SymmetricKeyGen(logger, keysDir, blockSize, params, rubato)
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
	logger.PrintHeader("[Server - Offline] Evaluates the keystreams (Eval^{FV}) to produce V")
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
		logger.PrintMessage("Halfboot X and outputs a CKKS-ciphertext M containing the CKKS encrypted messages in its slots")
		t = time.Now()
		ctBoot, _ = halfBootstrapper.HalfBoot(ciphertext, false)
		logger.PrintRunningTime("HalfBoot", t)

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
