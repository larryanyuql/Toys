import ase
import cv2
from ase.lattice.cubic import BodyCenteredCubic
from ase.io import write
import time
import numpy as np # import numpy for faster array operations


def delete_atoms(img, atoms, cut: int = 127):
    rows, cols = img.shape
    lattice = atoms.cell.cellpar()[2]
    undelete_id = []
    for id in range(len(atoms)):
        if atoms[id].position[2] != 0:
            undelete_id.append(id)
        else:
            i = int(atoms[id].position[1] / lattice)
            j = int(atoms[id].position[0] / lattice)
            if img[rows - i - 1, j] < cut:
                undelete_id.append(id)

    undelete_id = np.array(undelete_id)
    new_atoms = ase.Atoms(cell=atoms.get_cell(), pbc=atoms.get_pbc())
    new_atoms.extend(atoms[undelete_id])

    return new_atoms


def show_img_bin(_img):
    while True:
        img = _img.copy()
        cut = int(input("cut: 输入 0~255 的整数. (grayscale < cut ? 0 : 255): "))
        img[img < cut] = 0
        img[img >= cut] = 255
        cv2.imshow('IMAGE', img)
        cv2.waitKey()

        flag = input("是否合适？y/N: ")
        if flag == 'y':
            return img, cut


if __name__ == '__main__':
    start = time.perf_counter()

    # read an image file in grayscale mode (0) using cv2.imread()
    img = cv2.imread("93517999_p0.png", 0)
    # determine the value of cut
    # cut = 127
    img, cut = show_img_bin(img)

    rows, cols = img.shape
    # create an atoms object with a body-centered cubic structure using BodyCenteredCubic class
    atoms = BodyCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(cols, rows, 1), symbol='Fe', pbc=(1, 1, 1))
    # delete atoms where its grayscale < cut
    new_atoms = delete_atoms(img, atoms, cut)
    # write the new atoms object to a lammps-data file
    ase.io.write('atoms_fa.cfg', new_atoms, format='lammps-data')

    end = time.perf_counter()
    print('End!\nTotal time is ', end - start)
