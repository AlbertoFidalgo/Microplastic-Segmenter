import sys
import argparse

parser=argparse.ArgumentParser(description="segmentador/clasificador de microplasticos")
parser.add_argument("dir", type=str, help="directory to examine")
parser.add_argument("-o", type=str, help="directory to output, default is input dir")
parser.add_argument("--pps", type=int, default=64, help="points per side, default is 64")
args=parser.parse_args()

path = args.dir
output = path
if args.o:
    output = args.o
path = path.replace('\\', '/')
output = output.replace('\\', '/')
points_per_side = args.pps

print("Segmentador-Clasificador Inicializado!")
print("")

import segmentador_sam

wb = segmentador_sam.WhiteBalancer(3, 0.99)

up_path = ""
model = "edsr"
model_num = 2
#upscaler = segmentador_sam.Upscaler(2, up_path, model, model_num)

overlap_function = segmentador_sam.OverlapFunction(3)
separate_blobs = segmentador_sam.SeparateBlobs()

seg = segmentador_sam.Segmentador(1, whitebalancer=wb, upscaler=None, overlap_function=overlap_function, separate_blobs=separate_blobs, points_per_side=points_per_side)
clas = segmentador_sam.Clasificador("What colour is the stone?")

segclas = segmentador_sam.SegmentadorClasificador(path, seg, clas, output=output)
segclas.run()