package main

import (
	FLRubato "flhhe"
	"flhhe/configs"
	"flhhe/src/RtF"
	"flhhe/src/utils"
	"math"
	"path/filepath"
	"testing"
)

func TestLoadCipher(t *testing.T) {
	logger := utils.NewLogger(utils.DEBUG)
	root := FLRubato.FindRootPath()
	keysDir := filepath.Join(root, configs.Keys)
	ciphersDir := filepath.Join(root, configs.Ciphertexts)

	paramIndex := RtF.RUBATO128L
	// MFV instances
	var ckksEncoder RtF.CKKSEncoder
	var ckksDecryptor RtF.CKKSDecryptor
	var fvEvaluator RtF.MFVEvaluator

	// Rubato parameter
	plainModulus := RtF.RubatoParams[paramIndex].PlainModulus

	// RtF Rubato parameters, for full-coefficients only (128bit security)
	halfBsParams := RtF.RtFRubatoParams[0]
	params, err := halfBsParams.Params()
	if err != nil {
		panic(err)
	}
	params.SetPlainModulus(plainModulus)
	params.SetLogFVSlots(params.LogN())

	logger.PrintFormatted("PtX len: %d, %d", params.Slots(), params.N())

	// reading the already generated keys from a previous step, it will save time and memory :)
	_, ckksEncoder, _, ckksDecryptor, _, fvEvaluator = InitHHEScheme(logger, keysDir, params, halfBsParams)

	// wCt ciphertext for weights
	wCt := LoadCipher(logger, 0, ciphersDir, params)
	// bCt ciphertext for biases
	bCt := LoadCipher(logger, 1, ciphersDir, params)

	logger.PrintFormatted("degree %d", wCt.Degree())
	// resCt to store the results
	resCt := RtF.NewCiphertextFVLvl(params, 2, 0)
	resCt.Value()[0] = wCt.Value()[0].CopyNew() // to make sure they have the same degree

	// calculate the res = w*b
	fvEvaluator.Mul(wCt, bCt, resCt)

	wDec := ckksEncoder.DecodeComplex(ckksDecryptor.DecryptNew(wCt), params.LogSlots())
	bDec := ckksEncoder.DecodeComplex(ckksDecryptor.DecryptNew(bCt), params.LogSlots())
	resDec := ckksEncoder.DecodeComplex(ckksDecryptor.DecryptNew(resCt), params.LogSlots())

	logger.PrintFormatted("Level: %d (logQ = %d)", wCt.Level(), params.LogQLvl(wCt.Level()))
	logger.PrintFormatted("Scale: 2^%f", math.Log2(wCt.Scale()))

	logger.PrintFormatted("Weights{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(wDec), wDec[0], wDec[1], wDec[2], wDec[3])
	logger.PrintFormatted("Biases{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(bDec), bDec[0], bDec[1], bDec[2], bDec[3])
	logger.PrintFormatted("Results{%d}: [%6.10f %6.10f %6.10f %6.10f...]", len(resDec), resDec[0], resDec[1], resDec[2], resDec[3])
}

// ConvertCiphertext converts old Ciphertext to Lattigo v6 Ciphertext
//func ConvertCiphertext(oldCt *RtF.Ciphertext, params RtF.Parameters) *rlwe.Ciphertext {
//	// Create the new Ciphertext
//	newCt := &rlwe.Ciphertext{}
//
//	// Create Metadata
//	newCt.MetaData = &rlwe.MetaData{
//		PlaintextMetaData: rlwe.PlaintextMetaData{
//			Scale: rlwe.NewScaleModT(oldCt.Scale(), params.PlainModulus()),
//		},
//		CiphertextMetaData: rlwe.CiphertextMetaData{
//			IsNTT: oldCt.IsNTT(),
//			IsMontgomery: true,
//		},
//	}
//
//	// Convert []*ring.Poly to structs.Vector[ring.Poly]
//	t := make([]ring.Poly, 1)
//	newCt.Value = t
//	copy(newCt.Value, oldCt.Value())
//
//	return newCt
//}
