import numpy as np

def not_in(a, all):
    for i in all:
        if np.array_equal(i, a):
            return False
    return True

def bland_simplex(a, b, c, basis, nulls):
    all_basis = []
    optimal = False
    while not_in(basis, all_basis) and not optimal:
        print('B:', basis)
        print('N:', nulls)
        a_b = a[:, basis]
        a_b_inv = np.linalg.inv(a_b)
        print('Ab inv: \n', a_b_inv)
        x_b = np.dot(a_b_inv, b)

        print('Xb:', x_b)
        c_b = c[basis]
        cb_ab = np.dot(c_b, a_b_inv)

        reduced = np.array([c[j] - np.dot(cb_ab, a[:, j]) for j in nulls])
        print('Reduced Cost:', reduced)

        optimal = (reduced >= 0).all()

        if not optimal:
            entering = nulls[np.where(reduced < 0)[0][0]]

            d_b = np.dot(a_b_inv, a[:, entering])
            print('db:', d_b)
            positives = np.where(d_b > 0)[0]
            ratio = x_b[positives] / d_b[positives]
            print('ratio test:', ratio)
            leaving = basis[positives[np.argmin(ratio)]]
            print('entering:', entering)
            print('leaving:', leaving)

            all_basis += [basis]
            basis = np.sort(
                np.append(np.setdiff1d(basis, [leaving]),[entering])
            )
            nulls = np.sort(
                np.append(np.setdiff1d(nulls, [entering]), [leaving])
            )
            print('\n')

        if optimal:
            print('found optimal solution')

def simplex(a, b, c, basis, nulls):
    all_basis = []
    optimal = False

    while not_in(basis, all_basis) and not optimal:
        # input()
        print('B:', basis)
        print('N:', nulls)
        a_b = a[:, basis]
        a_b_inv = np.linalg.inv(a_b)
        print('Ab inv: \n', a_b_inv)

        x_b = np.dot(a_b_inv, b)
        print('Xb:', x_b)
        c_b = c[basis]
        cb_ab = np.dot(c_b, a_b_inv)
        print(cb_ab)

        reduced = np.array([c[j] - np.dot(cb_ab, a[:, j]) for j in nulls])
        optimal = (reduced <= 0).all()
        print('Reduced Cost:', reduced)

        entering = nulls[np.argmax(reduced)]
        d_b = np.dot(a_b_inv, a[:, entering])
        print('db:', d_b)
        positives = np.where(d_b > 0)[0]
        ratio = x_b[positives] / d_b[positives]
        print('ratio test:', ratio)
        leaving = basis[positives[np.argmin(ratio)]]
        print('entering:', entering)
        print('leaving:', leaving)

        all_basis += [basis]
        basis = np.sort(np.append(np.setdiff1d(basis, [leaving]), [entering]))
        nulls = np.sort(np.append(np.setdiff1d(nulls, [entering]), [leaving]))
        print('\n')

        if optimal:
            print('found optimal solution')

if __name__ == '__main__':

    a = np.array([1., -1., 1., 0., 1., 1., 0., 1.])
    a = np.reshape(a, (2, 4))
    b = np.array([2., 6.])
    c = np.array([-2., -1., 0., 0.])
    basis = np.array([2, 3])
    nulls = np.array([0, 1])

    bland_simplex(a, b, c, basis, nulls)

    a = np.array(
        [
            1., 0., 0., 1., 0., 0., 20., 1., 0., 0., 1., 0., 200., 20., 1., 0., 0., 1.
        ]
    )
    a = np.reshape(a, (3, 6))
    b = np.array([1, 100, 10000])
    c = np.array([100., 10., 1., 0., 0., 0.])
    basis = np.array([3, 4, 5])
    nulls = np.array([0, 1, 2])

    simplex(a, b, c, basis, nulls)