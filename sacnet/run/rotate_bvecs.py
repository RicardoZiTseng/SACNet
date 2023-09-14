# If you use this code, please cite our paper.
#
# Copyright (C) 2023 Zilong Zeng
# For any questions, please contact Dr.Zeng (zilongzeng@mail.bnu.edu.cn) or Dr.Zhao (tengdazhao@bnu.edu.cn).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import numpy as np

def read_matrix(matrix_file):
    with open(matrix_file) as f:
        lines = f.readlines()
    lines = lines[:3]
    elems = []
    for line in lines:
        line = [float(num) for num in line.split( )][:3]
        elems += line
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = elems
    return m11, m12, m13, m21, m22, m23, m31, m32, m33

def read_bvecs(bvec_file):
    with open(bvec_file) as f:
        lines = f.readlines()
    bvec_vals = []
    for line in lines:
        line = [float(num) for num in line.split( )]
        bvec_vals.append(line)
    bvec_vals = np.array(bvec_vals)
    bvec_X = bvec_vals[0, :]
    bvec_Y = bvec_vals[1, :]
    bvec_Z = bvec_vals[2, :]
    return bvec_X, bvec_Y, bvec_Z

def rotate_bvecs(bvec_file, matrix_file):
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = read_matrix(matrix_file)
    bvec_X, bvec_Y, bvec_Z = read_bvecs(bvec_file)
    bvec_X_rot = m11 * bvec_X + m12 * bvec_Y + m13 * bvec_Z
    bvec_Y_rot = m21 * bvec_X + m22 * bvec_Y + m23 * bvec_Z
    bvec_Z_rot = m31 * bvec_X + m32 * bvec_Y + m33 * bvec_Z
    return bvec_X_rot, bvec_Y_rot, bvec_Z_rot

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", required=True, help="Path to the input bvec file.")
    parse.add_argument("--matrix", required=True, help="Path to the FSL-fomat rotation matrix.")
    parse.add_argument("--output", required=True, help="Path to the output bvec file.")

    args = parse.parse_args()
    input = args.input
    matrix = args.matrix
    output = args.output

    bvec_X_rot, bvec_Y_rot, bvec_Z_rot = rotate_bvecs(input, matrix)

    bvec_X_rot = list(bvec_X_rot)
    bvec_X_rot = ["{:.6f}".format(b) for b in bvec_X_rot]
    bvec_X_rot = " ".join(bvec_X_rot)

    bvec_Y_rot = list(bvec_Y_rot)
    bvec_Y_rot = ["{:.6f}".format(b) for b in bvec_Y_rot]
    bvec_Y_rot = " ".join(bvec_Y_rot)

    bvec_Z_rot = list(bvec_Z_rot)
    bvec_Z_rot = ["{:.6f}".format(b) for b in bvec_Z_rot]
    bvec_Z_rot = " ".join(bvec_Z_rot)

    with open(output, mode='a') as f:
        f.write(bvec_X_rot)
        f.write("\n")
        f.write(bvec_Y_rot)
        f.write("\n")
        f.write(bvec_Z_rot)
        f.write("\n")
    
if __name__ == "__main__":
    main()
