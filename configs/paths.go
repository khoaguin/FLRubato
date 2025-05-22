package configs

// paths to plaintext and encrypted weights
const Configs = "configs/"
const PlaintextWeights = "weights/mnist/plain"
const DecryptedWeights = "weights/mnist/decrypted"
const SymmetricEncryptedWeights = "weights/mnist/symmetric_encrypted"
const HEEncryptedWeights = "weights/mnist/he_encrypted"

// Keys all the required keys to be stored
const Keys = "keys/keys128L/"
const SecretKey = "sk.bin"
const PublicKey = "pk.bin"
const StCDiagMatrix = "stcdm.bin"
const RotationKeys = "rot.bin"
const RelinearizationKeys = "re.bin"

const SymmetricKey = "symmetric_key.bin"
const SymmetricKeyCipherDir = "he_encrypted_symmetric_key"

//const HalfBootKeys = "hbst.bin"

// Ciphertexts the result CKKS ciphertexts after transciphering
const CtNameFix = "ctx_"
const CtFormat = ".bin"
