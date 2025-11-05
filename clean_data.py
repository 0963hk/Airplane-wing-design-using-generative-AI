import numpy as np
from io import StringIO
import os


def read_cartesian(data_path, clean_data=False):
    SEP = 0
    FIRST = 0

    xs = []
    ys = []
    with open(data_path) as f:
        for line in f:
            xy = np.genfromtxt(StringIO(line), dtype=float)
            if np.sum(np.isnan(xy)):
                continue
            if len(xy) == 2 and xy[0] - 1. < 1e-4 and xy[1] - 1. < 1e-4:
                xs.append(float(xy[0]))
                ys.append(float(xy[1]))
                if len(xs) == 2 and SEP == 0:
                    if xy[0] < xy_pre[0]:
                        SEP = 2
                    else:
                        SEP = 1
                if len(xs) > 1:
                    if SEP == 1 and xy[0] < xy_pre[0]:
                        xs0 = xs[:-1]
                        ys0 = ys[:-1]
                        xs = [xy[0]]
                        ys = [xy[1]]
                    if SEP == 2 and not FIRST and xy[0] > xy_pre[0]:
                        FIRST = 1
                        xs0 = xs[:-1]
                        ys0 = ys[:-1]
                        xs = [xy[0]]
                        ys = [xy[1]]
                xy_pre = xy

    if SEP == 1:
        xs0.reverse()
        ys0.reverse()

    xx = np.concatenate((xs0, xs), axis=0)
    yy = np.concatenate((ys0, ys), axis=0)
    Q = np.vstack((xx, yy)).transpose()

    D1 = np.diff(Q, n=1, axis=0)
    N1 = np.linalg.norm(D1, axis=1)
    ind1 = np.arange(N1.shape[0])[N1 == 0]
    Q = np.delete(Q, ind1 + 1, axis=0)

    D2 = Q[2:] - Q[:-2]
    N2 = np.linalg.norm(D2, axis=1)
    ind2 = np.arange(N2.shape[0])[N2 == 0]
    Q = np.delete(Q, ind2 + 2, axis=0)

    if clean_data:
        Q, del_ind = clean(Q)
        return Q, del_ind
    else:
        return Q


def clean(Q):
    D = np.diff(Q, n=1, axis=0)
    D2 = np.diff(D, n=1, axis=0)
    inner_prods = np.sum(D2[:-1] * D2[1:], axis=1)
    q_low, q_high = np.percentile(inner_prods, [10, 90])
    IQR = q_high - q_low
    low = q_low - 10 * IQR
    ind = np.arange(inner_prods.shape[0])[inner_prods < low]
    del_ind = []
    if len(ind) > 1:
        for i in range(len(ind) - 1):
            if ind[i] + 1 == ind[i + 1]:
                del_ind.append(ind[i + 1] + 1)
    Q = np.delete(Q, del_ind, axis=0)

    return Q, del_ind


def process_airfoils(input_folder='original_airfoils', output_folder='processed_airfoils', clean_data=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.dat') or filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                if clean_data:
                    processed_data, _ = read_cartesian(input_path, clean_data=True)
                else:
                    processed_data = read_cartesian(input_path, clean_data=False)

                np.savetxt(output_path, processed_data, fmt='%.6f')
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    process_airfoils()
