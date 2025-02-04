package main

import (
	"crypto/rand"
	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/utils"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strconv"
	"time"
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

	mws := make([]utils.ModelWeights, 3)
	mws[0], mws[1], mws[2] = OpenModelWeights(logger, root)

	for _, mw := range mws {
		mw.Print2DLayerDimension(logger)
	}

	// Always use fullCoffs = true
	Rubato(logger, root, RtF.RUBATO128L, mws, true)
}

// OpenModelWeights read the model weights from
func OpenModelWeights(logger utils.Logger, root string) (utils.ModelWeights, utils.ModelWeights, utils.ModelWeights) {
	var err error
	weightDir := filepath.Join(root, configs.MNIST)
	logger.PrintHeader("FLClient: Load plaintext weights from JSON (after training in python)")
	w1 := utils.NewModelWeights()
	err = w1.LoadWeights(weightDir + "/mnist_weights_exclude_137.json")
	utils.HandleError(err)

	w2 := utils.NewModelWeights()
	err = w2.LoadWeights(weightDir + "/mnist_weights_exclude_258.json")
	utils.HandleError(err)

	w3 := utils.NewModelWeights()
	err = w3.LoadWeights(weightDir + "/mnist_weights_exclude_469.json")
	utils.HandleError(err)

	return w1, w2, w3
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
	blockSize := RtF.RubatoParams[paramIndex].Blocksize
	outputSize := blockSize - 4
	numRound := RtF.RubatoParams[paramIndex].NumRound
	plainModulus := RtF.RubatoParams[paramIndex].PlainModulus
	sigma := RtF.RubatoParams[paramIndex].Sigma

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
	} else {
		params.SetLogFVSlots(params.LogSlots())
	}

	// !!!
	// NOTE: We call this function only once to generate the keys and store them in files
	//InitialKeyGen(logger, keysDir, params, halfBsParams, fullCoffs)

	// reading the already generated keys from a previous step, it will save time and memory :)
	fvEncoder, ckksEncoder, fvEncryptor, ckksDecryptor, halfBootstrapper, fvEvaluator = Setup(logger, keysDir, params, halfBsParams)

	// Allocating the coefficients
	coefficients := make([][]float64, outputSize)
	for s := 0; s < outputSize; s++ {
		coefficients[s] = make([]float64, params.N())
	}

	// Key generation
	var key []uint64
	key = make([]uint64, blockSize)
	for i := 0; i < blockSize; i++ {
		key[i] = uint64(i + 1) // Use (1, ..., 16) for testing
	}

	var data [][]float64
	var nonces [][]byte
	var counter []byte
	var keystream [][]uint64
	var fvKeyStreams []*RtF.Ciphertext

	if fullCoffs {
		logger.PrintHeader("Let's make the data usable! (full coefficients)")
		data = preparingData(logger, outputSize, params, mws)
		nonces = make([][]byte, params.N())
		for i := 0; i < params.N(); i++ {
			nonces[i] = make([]byte, 64)
			rand.Read(nonces[i])
		}
		counter = make([]byte, 64)
		rand.Read(counter)

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

		plainCKKSRingTs = make([]*RtF.PlaintextRingT, outputSize)
		for s := 0; s < outputSize; s++ {
			plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coefficients[s], messageScaling)
			poly := plainCKKSRingTs[s].Value()[0]
			for i := 0; i < params.N(); i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
				poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus()
			}
		}
	} else {
		data = make([][]float64, outputSize)
		for s := 0; s < outputSize; s++ {
			data[s] = make([]float64, params.Slots())
			for i := 0; i < params.Slots(); i++ {
				data[s][i] = utils.RandFloat64(-1, 1)
			}
		}

		nonces = make([][]byte, params.Slots())
		for i := 0; i < params.Slots(); i++ {
			nonces[i] = make([]byte, 64)
			rand.Read(nonces[i])
		}
		counter = make([]byte, 64)
		rand.Read(counter)

		keystream = make([][]uint64, params.Slots())
		for i := 0; i < params.Slots(); i++ {
			keystream[i] = RtF.PlainRubato(blockSize, numRound, nonces[i], counter, key, params.PlainModulus(), sigma)
		}

		for s := 0; s < outputSize; s++ {
			for i := 0; i < params.Slots()/2; i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()-1))
				coefficients[s][j] = data[s][i]
				coefficients[s][j+uint64(params.N()/2)] = data[s][i+params.Slots()/2]
			}
		}

		// encrypt the plaintext data using the symmetric key stream
		plainCKKSRingTs = make([]*RtF.PlaintextRingT, outputSize)
		for s := 0; s < outputSize; s++ {
			plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coefficients[s], messageScaling)
			poly := plainCKKSRingTs[s].Value()[0]
			for i := 0; i < params.Slots(); i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
				poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus()
			}
		}
	}

	plaintexts = make([]*RtF.Plaintext, outputSize)
	for s := 0; s < outputSize; s++ {
		plaintexts[s] = RtF.NewPlaintextFVLvl(params, 0)
		fvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}
	logger.PrintMemUsage("PtScaleUp")

	// FV Keystream
	rubato := RtF.NewMFVRubato(paramIndex, params, fvEncoder, fvEncryptor, fvEvaluator, rubatoModDown[0])
	kCt := rubato.EncKey(key)
	logger.PrintMemUsage("EncSymKey")

	//fvKeyStreams = rubato.Crypt(nonces, counter, kCt, rubatoModDown)
	fvKeyStreams = rubato.CryptNoModSwitch(nonces, counter, kCt)
	for i := 0; i < outputSize; i++ {
		fvKeyStreams[i] = fvEvaluator.SlotsToCoeffs(fvKeyStreams[i], stcModDown)
		fvEvaluator.ModSwitchMany(fvKeyStreams[i], fvKeyStreams[i], fvKeyStreams[i].Level())
	}
	logger.PrintMemUsage("FvKeyStreams")

	var ctBoot *RtF.Ciphertext
	outputSize = 1 // for testing
	for s := 0; s < outputSize; s++ {
		// Encrypt and mod switch to the lowest level
		ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
		ciphertext.Value()[0] = plaintexts[s].Value()[0].CopyNew()
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
			ctBoot, _ = halfBootstrapper.HalfBoot(ciphertext, false)
		} else {
			ctBoot, _ = halfBootstrapper.HalfBoot(ciphertext, true)
		}
		logger.PrintMemUsage("HalfBoot")

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

func preparingData(logger utils.Logger, outputSize int, params *RtF.Parameters, mws []utils.ModelWeights) [][]float64 {
	data := make([][]float64, outputSize)
	//for s := 0; s < outputSize; s++ {
	//	data[s] = make([]float64, params.N())
	//	for i := 0; i < params.N(); i++ {
	//		data[s][i] = utils.RandFloat64(-1, 1)
	//	}
	//}

	logger.PrintFormatted("The data structure is as [%d][%d].", outputSize, params.N())
	logger.PrintFormatted("We have the flatten weights as [%d] and [%d]", len(mws[0].FC1Flatten), len(mws[0].FC2Flatten))

	cnt := 0 // will use this counter for locating
	for _, mw := range mws {
		// start with FC1
		cipherPerFC1 := int(math.Ceil(float64(len(mw.FC1Flatten)) / float64(params.N())))
		paddingLenFC1 := params.N() - (len(mw.FC1Flatten) / cipherPerFC1)
		fc1Space := params.N() - paddingLenFC1
		logger.PrintFormatted("Number of ciphers required to store FC1: %d", cipherPerFC1)
		logger.PrintFormatted("Padding length required to store FC1: %d", paddingLenFC1)
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

	}

	// filling the rest with padding (this is not efficient at all) -> the solution will be changing the parameters
	for s := cnt; s < outputSize; s++ {
		data[s] = make([]float64, params.N())
		for i := 0; i < params.N(); i++ {
			data[s][i] = float64(0)
		}
	}
	return data
}

// SaveCipher save a ciphertext in the provided path
func SaveCipher(logger utils.Logger, index int, ciphersDir string, ciphertext *RtF.Ciphertext) {
	logger.PrintHeader("Save the CKKS ciphertext in a file for further computation")
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(index) + configs.CtFormat
	err = utils.Serialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
}

// LoadCipher save a ciphertext in the provided path
func LoadCipher(logger utils.Logger, fileIndex int, ciphersDir string, params *RtF.Parameters) *RtF.Ciphertext {
	logger.PrintHeader("Load the CKKS ciphertext from the file")
	var err error
	fileName := configs.CtNameFix + strconv.Itoa(fileIndex) + configs.CtFormat
	ciphertext := RtF.NewCiphertextFVLvl(params, 1, 0)
	err = utils.Deserialize(ciphertext, filepath.Join(ciphersDir, fileName))
	utils.HandleError(err)
	return ciphertext
}

// InitialKeyGen we need to run this function only once, so the CPS generate and store the keys and pp
func InitialKeyGen(
	logger utils.Logger,
	keysDir string,
	params *RtF.Parameters,
	hbtParams *RtF.HalfBootParameters,
	fullCoffs bool) {
	logger.PrintHeader("Initializing the generation of keys and public parameters")
	var err error

	kgen := RtF.NewKeyGenerator(params)
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

	return
}

func Setup(
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
