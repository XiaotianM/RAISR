import argparse

def getArgs():
    parser = argparse.ArgumentParser(description="args for RAISR")
    parser.add_argument("--rate", type=int, default=2, help="upsacling factor default x2")
    parser.add_argument("--patchSize", type=int, default=11, help="image patch size")
    parser.add_argument("--neigborSize", type=int, default=9, help="the neighborhood size for computation key for hash")
    parser.add_argument("--stride", type=int, default=1, help="control overlap for patches(control K samples)")

    parser.add_argument("--Qangle", type=int, default=24, help="quantization factor of angle")
    parser.add_argument("--Qstrength", type=int, default=8, help="quantization factor of strength")
    parser.add_argument("--Qcoherence", type=int, default=8, help="quantization factor of coherence")
    
    parser.add_argument("--train_dataset", type=str, default="./datasets/T91/", help="Training dataSet path")
    parser.add_argument("--test_dataset", type=str, default="./datasets/Set5/", help="Test dataSets")

    args = parser.parse_args()
    return args