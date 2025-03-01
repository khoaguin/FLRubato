package client

import (
	"crypto/rand"
	"log"
	"math"
	"path/filepath"

	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/hhe_fedavg/keys_dealer"
	"flhhe/src/utils"
)

func RunFLClient(
	logger utils.Logger,
	rootPath string,
	params *keys_dealer.RubatoParams,
	weightPath string,
) {
	logger.PrintHeader("--- Client ---")
	logger.PrintHeader("[Client - Initialization]: Load plaintext weights from JSON")
	
	keysDir := filepath.Join(rootPath, configs.Keys)
	
	modelWeights := utils.OpenModelWeights(logger, rootPath, weightPath)
	modelWeights.Print2DLayerDimension(logger)

	logger.PrintHeader("[Client] Preparing the data")
	var data [][]float64 = PreparingData(logger, 3, params.Params, modelWeights)
	logger.PrintFormatted("Data.shape = [%d][%d]", len(data), len(data[0]))


	logger.PrintHeader("[Client - Offline] Generating the nonces and counter")
	nonces := make([][]byte, params.Params.N())
	for i := 0; i < params.Params.N(); i++ {
		nonces[i] = make([]byte, 64)
		rand.Read(nonces[i])
	}
	logger.PrintFormatted("Nonces diminsion: [%d][%d]", len(nonces), len(nonces[0]))
	counter := make([]byte, 64)
	rand.Read(counter)
	logger.PrintFormatted("Counter diminsion: [%d]", len(counter))

	logger.PrintHeader("[Client - Offline] Loading the symmetric key")
	symKeyPath := filepath.Join(keysDir, configs.SymmetricKey)
	symKey := keys_dealer.LoadSymmKey(symKeyPath, params.Blocksize)

	logger.PrintHeader("[Client - Offline] Generating the keystream z")
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

	// for s := 0; s < outputSize; s++ {
	// 	for i := 0; i < params.N()/2; i++ {
	// 		j := utils.BitReverse64(uint64(i), uint64(params.Params.LogN()-1))
	// 		coefficients[s][j] = data[s][i]
	// 		coefficients[s][j+uint64(params.Params.N()/2)] = data[s][i+params.Params.N()/2]
	// 	}
	// }

	// logger.PrintHeader("[Client - Online] Encrypting the plaintext data using the symmetric key stream")
	// plainCKKSRingTs = make([]*RtF.PlaintextRingT, outputSize)
	// for s := 0; s < outputSize; s++ {
	// 	plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coefficients[s], messageScaling) // scales up the plaintext message
	// 	poly := plainCKKSRingTs[s].Value()[0]
	// 	for i := 0; i < params.N(); i++ {
	// 		j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
	// 		poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus() // modulo q addition between the keystream to the scaled message
	// 	}
	// }
}

func PreparingData(logger utils.Logger, outputSize int, params *RtF.Parameters, mw utils.ModelWeights) [][]float64 {
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
	logger.PrintFormatted("Number of ciphers required to store FC1 = len(FC1Flatten) / params.N(): %d", cipherPerFC1)
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
