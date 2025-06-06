package RtF

import (
	"crypto/rand"
	"fmt"
	"math"
	"testing"

	"flhhe/src/utils"
)

// Benchmark RtF framework with HERA for 80-bit security full-slots parameter
func BenchmarkRtFHera80f(b *testing.B) {
	benchmarkRtFHera(b, "80f", 4, 0, 2, true)
}

// Benchmark RtF framework with HERA for 80-bit security 4-slots parameter
func BenchmarkRtFHera80s(b *testing.B) {
	benchmarkRtFHera(b, "80s", 4, 1, 0, false)
}

// Benchmark RtF framework with HERA for 80-bit security full-slots parameter with arcsine evaluation
func BenchmarkRtFHera80af(b *testing.B) {
	benchmarkRtFHera(b, "80af", 4, 2, 2, true)
}

// Benchmark RtF framework with HERA for 80-bit security 4-slots parameter with arcsine evaluation
func BenchmarkRtFHera80as(b *testing.B) {
	benchmarkRtFHera(b, "80as", 4, 3, 0, false)
}

// Benchmark RtF framework with HERA for 128-bit security full-slots parameter
func BenchmarkRtFHera128f(b *testing.B) {
	benchmarkRtFHera(b, "128f", 5, 0, 2, true)
}

// Benchmark RtF framework with HERA for 128-bit security 4-slots parameter
func BenchmarkRtFHera128s(b *testing.B) {
	benchmarkRtFHera(b, "128s", 5, 1, 0, false)
}

// Benchmark RtF framework with HERA for 128-bit security full-slots parameter with arcsine evaluation
func BenchmarkRtFHera128af(b *testing.B) {
	benchmarkRtFHera(b, "128af", 5, 2, 2, true)
}

// Benchmark RtF framework with HERA for 128-bit security 4-slots parameter with arcsine evaluation
func BenchmarkRtFHera128as(b *testing.B) {
	benchmarkRtFHera(b, "128as", 5, 3, 2, false)
}

// Benchmark RtF framework with Rubato80S
func BenchmarkRtFRubato80S(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO80S)
}

// Benchmark RtF framework with Rubato80M
func BenchmarkRtFRubato80M(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO80M)
}

// Benchmark RtF framework with Rubato80L
func BenchmarkRtFRubato80L(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO80L)
}

// Benchmark RtF framework with Rubato128S
func BenchmarkRtFRubato128S(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO128S)
}

// Benchmark RtF framework with Rubato128M
func BenchmarkRtFRubato128M(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO128M)
}

// Benchmark RtF framework with Rubato128L
func BenchmarkRtFRubato128L(b *testing.B) {
	benchmarkRtFRubato(b, RUBATO128L)
}

func benchmarkRtFHera(b *testing.B, name string, numRound int, paramIndex int, radix int, fullCoeffs bool) {
	var err error

	var hbtp *HalfBootstrapper
	var kgen KeyGenerator
	var fvEncoder MFVEncoder
	var ckksEncoder CKKSEncoder
	var ckksDecryptor CKKSDecryptor
	var sk *SecretKey
	var pk *PublicKey
	var fvEncryptor MFVEncryptor
	var fvEvaluator MFVEvaluator
	var plainCKKSRingTs []*PlaintextRingT
	var plaintexts []*Plaintext
	var hera MFVHera

	var data [][]float64
	var nonces [][]byte
	var key []uint64
	var keystream [][]uint64
	var fvKeystreams []*Ciphertext

	// RtF Hera parameters
	// Four sets of parameters (index 0 to 3) ensuring 128 bit of security
	// are available in github.com/smilecjf/lattigo/v2/ckks_fv/rtf_params
	// LogSlots is hardcoded in the parameters, but can be changed from 4 to 15.
	// When changing logSlots make sure that the number of levels allocated to CtS is
	// smaller or equal to logSlots.

	hbtpParams := RtFHeraParams[paramIndex]
	params, err := hbtpParams.Params()
	if err != nil {
		panic(err)
	}
	messageScaling := float64(params.PlainModulus()) / hbtpParams.MessageRatio

	// HERA parameters in RtF
	var heraModDown, stcModDown []int
	if numRound == 4 {
		heraModDown = HeraModDownParams80[paramIndex].CipherModDown
		stcModDown = HeraModDownParams80[paramIndex].StCModDown
	} else {
		heraModDown = HeraModDownParams128[paramIndex].CipherModDown
		stcModDown = HeraModDownParams128[paramIndex].StCModDown
	}

	// fullCoeffs denotes whether full coefficients are used for data encoding
	if fullCoeffs {
		params.SetLogFVSlots(params.LogN())
	} else {
		params.SetLogFVSlots(params.LogSlots())
	}

	// Scheme context and keys
	kgen = NewKeyGenerator(params)

	sk, pk = kgen.GenKeyPairSparse(hbtpParams.H)

	fvEncoder = NewMFVEncoder(params)
	ckksEncoder = NewCKKSEncoder(params)
	fvEncryptor = NewMFVEncryptorFromPk(params, pk)
	ckksDecryptor = NewCKKSDecryptor(params, sk)

	// Generating half-bootstrapping keys
	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(params.LogSlots(), hbtpParams)
	pDcds := fvEncoder.GenSlotToCoeffMatFV(radix)
	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(pDcds)
	rotations := append(rotationsHalfBoot, rotationsStC...)
	if !fullCoeffs {
		rotations = append(rotations, params.Slots()/2)
	}
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	rlk := kgen.GenRelinearizationKey(sk)
	hbtpKey := BootstrappingKey{Rlk: rlk, Rtks: rotkeys}

	if hbtp, err = NewHalfBootstrapper(params, hbtpParams, hbtpKey); err != nil {
		panic(err)
	}

	// Encode float data added by keystream to plaintext coefficients
	fvEvaluator = NewMFVEvaluator(params, EvaluationKey{Rlk: rlk, Rtks: rotkeys}, pDcds)
	coeffs := make([][]float64, 16)
	for s := 0; s < 16; s++ {
		coeffs[s] = make([]float64, params.N())
	}

	key = make([]uint64, 16)
	for i := 0; i < 16; i++ {
		key[i] = uint64(i + 1) // Use (1, ..., 16) for testing
	}

	if fullCoeffs {
		data = make([][]float64, 16)
		for s := 0; s < 16; s++ {
			data[s] = make([]float64, params.N())
			for i := 0; i < params.N(); i++ {
				data[s][i] = utils.RandFloat64(-1, 1)
			}
		}

		nonces = make([][]byte, params.N())
		for i := 0; i < params.N(); i++ {
			nonces[i] = make([]byte, 64)
			rand.Read(nonces[i])
		}

		keystream = make([][]uint64, params.N())
		for i := 0; i < params.N(); i++ {
			keystream[i] = PlainHera(numRound, nonces[i], key, params.PlainModulus())
		}

		for s := 0; s < 16; s++ {
			for i := 0; i < params.N()/2; i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()-1))
				coeffs[s][j] = data[s][i]
				coeffs[s][j+uint64(params.N()/2)] = data[s][i+params.N()/2]
			}
		}

		plainCKKSRingTs = make([]*PlaintextRingT, 16)
		for s := 0; s < 16; s++ {
			plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coeffs[s], messageScaling)
			poly := plainCKKSRingTs[s].Value()[0]
			for i := 0; i < params.N(); i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
				poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus()
			}
		}
	} else {
		data = make([][]float64, 16)
		for s := 0; s < 16; s++ {
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

		keystream = make([][]uint64, params.Slots())
		for i := 0; i < params.Slots(); i++ {
			keystream[i] = PlainHera(numRound, nonces[i], key, params.PlainModulus())
		}

		for s := 0; s < 16; s++ {
			for i := 0; i < params.Slots()/2; i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()-1))
				coeffs[s][j] = data[s][i]
				coeffs[s][j+uint64(params.N()/2)] = data[s][i+params.Slots()/2]
			}
		}

		plainCKKSRingTs = make([]*PlaintextRingT, 16)
		for s := 0; s < 16; s++ {
			plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coeffs[s], messageScaling)
			poly := plainCKKSRingTs[s].Value()[0]
			for i := 0; i < params.Slots(); i++ {
				j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
				poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus()
			}
		}

	}

	plaintexts = make([]*Plaintext, 16)

	for s := 0; s < 16; s++ {
		plaintexts[s] = NewPlaintextFVLvl(params, 0)
		fvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}

	hera = NewMFVHera(numRound, params, fvEncoder, fvEncryptor, fvEvaluator, heraModDown[0])
	kCt := hera.EncKey(key)

	// FV Keystream
	benchOffLat := fmt.Sprintf("RtF HERA Offline Latency")
	b.Run(benchOffLat, func(b *testing.B) {
		fvKeystreams = hera.Crypt(nonces, kCt, heraModDown)
		for i := 0; i < 1; i++ {
			fvKeystreams[i] = fvEvaluator.SlotsToCoeffs(fvKeystreams[i], stcModDown)
			fvEvaluator.ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		}
	})
	/* We assume that b.N == 1 */
	benchOffThrput := fmt.Sprintf("RtF HERA Offline Throughput")
	b.Run(benchOffThrput, func(b *testing.B) {
		for i := 1; i < 16; i++ {
			fvKeystreams[i] = fvEvaluator.SlotsToCoeffs(fvKeystreams[i], stcModDown)
			fvEvaluator.ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		}
	})

	var ctBoot *Ciphertext
	benchOnline := fmt.Sprintf("RtF HERA Online Lat x1")
	b.Run(benchOnline, func(b *testing.B) {
		// Encrypt and mod switch to the lowest level
		ciphertext := NewCiphertextFVLvl(params, 1, 0)
		ciphertext.Value()[0] = plaintexts[0].Value()[0].CopyNew()
		fvEvaluator.Sub(ciphertext, fvKeystreams[0], ciphertext)
		fvEvaluator.TransformToNTT(ciphertext, ciphertext)
		ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(params.Qi()[0]) / float64(params.PlainModulus()) * messageScaling))))

		// Half-Bootstrap the ciphertext (homomorphic evaluation of ModRaise -> SubSum -> CtS -> EvalMod)
		// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
		// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
		// Difference from the bootstrapping is that the last StC is missing.
		// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
		// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
		if fullCoeffs {
			ctBoot, _ = hbtp.HalfBoot(ciphertext, false)
		} else {
			ctBoot, _ = hbtp.HalfBoot(ciphertext, true)
		}
	})
	valuesWant := make([]complex128, params.Slots())
	for i := 0; i < params.Slots(); i++ {
		valuesWant[i] = complex(data[0][i], 0)
	}

	fmt.Println("Precision of HalfBoot(ciphertext)")
	printDebug(params, ctBoot, valuesWant, ckksDecryptor, ckksEncoder)
}

func benchmarkRtFRubato(b *testing.B, rubatoParam int) {
	var err error

	var hbtp *HalfBootstrapper
	var kgen KeyGenerator
	var fvEncoder MFVEncoder
	var ckksEncoder CKKSEncoder
	var ckksDecryptor CKKSDecryptor
	var sk *SecretKey
	var pk *PublicKey
	var fvEncryptor MFVEncryptor
	var fvEvaluator MFVEvaluator
	var plainCKKSRingTs []*PlaintextRingT
	var plaintexts []*Plaintext
	var rubato MFVRubato

	var data [][]float64
	var nonces [][]byte
	var counter []byte
	var key []uint64
	var keystream [][]uint64
	var fvKeystreams []*Ciphertext

	// Rubato parameter
	blocksize := RubatoParams[rubatoParam].Blocksize
	outputsize := blocksize - 4
	numRound := RubatoParams[rubatoParam].NumRound
	plainModulus := RubatoParams[rubatoParam].PlainModulus
	sigma := RubatoParams[rubatoParam].Sigma

	// RtF Rubato parameters
	hbtpParams := RtFRubatoParams[0]
	params, err := hbtpParams.Params()
	if err != nil {
		panic(err)
	}
	params.SetPlainModulus(plainModulus)
	params.SetLogFVSlots(params.LogN())
	messageScaling := float64(params.PlainModulus()) / hbtpParams.MessageRatio

	rubatoModDown := RubatoModDownParams[rubatoParam].CipherModDown
	stcModDown := RubatoModDownParams[rubatoParam].StCModDown

	// Scheme context and keys
	kgen = NewKeyGenerator(params)
	sk, pk = kgen.GenKeyPairSparse(hbtpParams.H)

	fvEncoder = NewMFVEncoder(params)
	ckksEncoder = NewCKKSEncoder(params)
	fvEncryptor = NewMFVEncryptorFromPk(params, pk)
	ckksDecryptor = NewCKKSDecryptor(params, sk)

	// Generating half-bootstrapping keys
	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(params.LogSlots(), hbtpParams)
	pDcds := fvEncoder.GenSlotToCoeffMatFV(2) // radix = 2 // needed to be store
	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(pDcds)
	rotations := append(rotationsHalfBoot, rotationsStC...)
	// needed to be saved
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	rlk := kgen.GenRelinearizationKey(sk)
	hbtpKey := BootstrappingKey{Rlk: rlk, Rtks: rotkeys}

	if hbtp, err = NewHalfBootstrapper(params, hbtpParams, hbtpKey); err != nil {
		panic(err)
	}

	// Encode float data added by key stream to plaintext coefficients
	fvEvaluator = NewMFVEvaluator(params, EvaluationKey{Rlk: rlk, Rtks: rotkeys}, pDcds)
	coeffs := make([][]float64, outputsize)
	for s := 0; s < outputsize; s++ {
		coeffs[s] = make([]float64, params.N())
	}

	// Key generation
	key = make([]uint64, blocksize)
	for i := 0; i < blocksize; i++ {
		key[i] = uint64(i + 1) // Use (1, ..., 16) for testing
	}

	// Get random data in [-1, 1]
	data = make([][]float64, outputsize)
	for s := 0; s < outputsize; s++ {
		data[s] = make([]float64, params.N())
		for i := 0; i < params.N(); i++ {
			data[s][i] = utils.RandFloat64(-1, 1)
		}
	}

	nonces = make([][]byte, params.N())
	for i := 0; i < params.N(); i++ {
		nonces[i] = make([]byte, 8)
		rand.Read(nonces[i])
	}
	counter = make([]byte, 8)
	rand.Read(counter)

	// Get keystream
	keystream = make([][]uint64, params.N())
	for i := 0; i < params.N(); i++ {
		keystream[i] = PlainRubato(blocksize, numRound, nonces[i], counter, key, plainModulus, sigma)
	}

	for s := 0; s < outputsize; s++ {
		for i := 0; i < params.N()/2; i++ {
			j := utils.BitReverse64(uint64(i), uint64(params.LogN()-1))
			coeffs[s][j] = data[s][i]
			coeffs[s][j+uint64(params.N()/2)] = data[s][i+params.N()/2]
		}
	}

	// Encode plaintext and Encrypt with key stream
	plainCKKSRingTs = make([]*PlaintextRingT, outputsize)
	for s := 0; s < outputsize; s++ {
		plainCKKSRingTs[s] = ckksEncoder.EncodeCoeffsRingTNew(coeffs[s], messageScaling)
		poly := plainCKKSRingTs[s].Value()[0]
		for i := 0; i < params.N(); i++ {
			j := utils.BitReverse64(uint64(i), uint64(params.LogN()))
			// encrypt data symmetrically
			poly.Coeffs[0][j] = (poly.Coeffs[0][j] + keystream[i][s]) % params.PlainModulus()
		}
	}

	plaintexts = make([]*Plaintext, outputsize)
	for s := 0; s < outputsize; s++ {
		plaintexts[s] = NewPlaintextFVLvl(params, 0)
		fvEncoder.FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}

	// FV Keystream
	rubato = NewMFVRubato(rubatoParam, params, fvEncoder, fvEncryptor, fvEvaluator, rubatoModDown[0])
	kCt := rubato.EncKey(key)

	benchOffLat := fmt.Sprintf("RtF Rubato Offline Latency")
	b.Run(benchOffLat, func(b *testing.B) {
		fvKeystreams = rubato.Crypt(nonces, counter, kCt, rubatoModDown)
		for i := 0; i < 1; i++ {
			fvKeystreams[i] = fvEvaluator.SlotsToCoeffs(fvKeystreams[i], stcModDown)
			fvEvaluator.ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		}
	})
	/* We assume that b.N == 1 */
	benchOffThrput := fmt.Sprintf("RtF Rubato Offline Throughput")
	b.Run(benchOffThrput, func(b *testing.B) {
		for i := 1; i < outputsize; i++ {
			fvKeystreams[i] = fvEvaluator.SlotsToCoeffs(fvKeystreams[i], stcModDown)
			fvEvaluator.ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		}
	})

	var ctBoot *Ciphertext
	benchOnline := fmt.Sprintf("RtF Rubato Online Lat x1")
	b.Run(benchOnline, func(b *testing.B) {
		// Encrypt and mod switch to the lowest level
		ciphertext := NewCiphertextFVLvl(params, 1, 0)
		ciphertext.Value()[0] = plaintexts[0].Value()[0].CopyNew()
		fvEvaluator.Sub(ciphertext, fvKeystreams[0], ciphertext)
		fvEvaluator.TransformToNTT(ciphertext, ciphertext)
		ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(params.Qi()[0]) / float64(params.PlainModulus()) * messageScaling))))

		// Half-Bootstrap the ciphertext (homomorphic evaluation of ModRaise -> SubSum -> CtS -> EvalMod)
		// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
		// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
		// Difference from the bootstrapping is that the last StC is missing.
		// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
		// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
		ctBoot, _ = hbtp.HalfBoot(ciphertext, false)
	})
	valuesWant := make([]complex128, params.Slots())
	for i := 0; i < params.Slots(); i++ {
		valuesWant[i] = complex(data[0][i], 0)
	}

	fmt.Println("Precision of HalfBoot(ciphertext)")
	printDebug(params, ctBoot, valuesWant, ckksDecryptor, ckksEncoder)
}

func printDebug(params *Parameters, ciphertext *Ciphertext, valuesWant []complex128, decryptor CKKSDecryptor, encoder CKKSEncoder) {

	valuesTest := encoder.DecodeComplex(decryptor.DecryptNew(ciphertext), params.LogSlots())
	logSlots := params.LogSlots()
	sigma := params.Sigma()

	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
	fmt.Printf("ValuesWant: %6.10f %6.10f %6.10f %6.10f...\n", valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

	precStats := GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, logSlots, sigma)

	fmt.Println(precStats.String())
}
