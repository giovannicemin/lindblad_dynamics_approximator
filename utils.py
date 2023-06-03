import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
from itertools import product


def get_params_from_cmdline(argv, default_params=None):
    '''Function that parse command line argments to dicitonary
    Parameters
    ----------
    argv : argv from sys.argv
    default_params : dict
        Dicionary to update

    Return
    ------
    Updated dictionary as in input
    '''
    arg_help = '{0} -L <length of spin chain> -b <beta> -V <potential> -w <working dir>'

    if default_params == None:
        raise Exception('Missing default parameters')

    try:
        opts, args = getopt.getopt(argv[1:], 'hL:b:V:w:', ['help', 'length', 'beta=', 'potential=', 'working_dir='])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(arg_help)
        elif opt in ('-L', '--length'):
            default_params['L'] = int(arg)
        elif opt in ('-b', '--beta'):
            default_params['beta'] = float(arg)
        elif opt in ('-p', '--potential'):
            default_params['potential'] = float(arg)
        elif opt in ('-w', '--working_dir'):
            default_params['working_dir'] = arg

    return default_params


def print_comparison(data_tebd, data_ml, T, dt, ylim):
    '''Function to print the comparison btween tebd data and ml prediction.
    Parameters
    ----------
    data_tebd : array
        data from the TEBD simulation
    data_ml : array
        data representing the ml algorithm predcition
    T : float
        total time of the evolution
    dt : float
        interval of time between data components
    ylim : float
        setting the y-axis limit [-ylim, ylim]
    '''
    t = np.arange(0, T, dt)[:-1]

    rows= 3
    columns = 5

    names = [r'$ \langle \sigma^x_2 \rangle /2$',
             r'$ \langle \sigma^y_2 \rangle /2$',
             r'$ \langle \sigma^z_2 \rangle /2$',
             r'$ \langle \sigma^x_1 \rangle /2$',
             r'$ \langle \sigma^x_1 \sigma^x_2 \rangle /2$',
             r'$ \langle \sigma^x_1 \sigma^y_2 \rangle /2$',
             r'$ \langle \sigma^x_1 \sigma^z_2 \rangle /2$',
             r'$ \langle \sigma^y_1 \rangle /2$',
             r'$ \langle \sigma^y_1 \sigma^x_2 \rangle /2$',
             r'$ \langle \sigma^y_1 \sigma^y_2 \rangle /2$',
             r'$ \langle \sigma^y_1 \sigma^z_2 \rangle /2$',
             r'$ \langle \sigma^z_1 \rangle /2$',
             r'$ \langle \sigma^z_1 \sigma^x_2 \rangle /2$',
             r'$ \langle \sigma^z_1 \sigma^y_2 \rangle /2$',
             r'$ \langle \sigma^z_1 \sigma^z_2 \rangle /2$',
             ]

    fig, axs = plt.subplots(rows, columns, figsize=(15,8), dpi=100)
    plt.setp(axs, xlim=(0,T), ylim=(-ylim, ylim))

    for i in range(rows):
        for j in range(columns):
            axs[i, j].plot(t, [data_tebd[k][columns*i+j] for k in range(len(t))], label='Simulation', color='k')
            axs[i, j].plot(t, [data_ml[k][columns*i+j] for k in range(len(t))], label='ml', color='r', linestyle='--')
            # title
            axs[i, j].set_title(names[columns*i+j], x=0.5, y=1.1, fontsize=20)
            # setting the ticks
            axs[i, j].tick_params(axis='both', which='both', direction='in', top=True, right=True)
            axs[i, j].tick_params(axis='y', labelsize=20)
            axs[i, j].tick_params(axis='x', labelsize=20)

            if i != rows-1:
                axs[i,j].set_xticklabels([])
            if j != 0:
                axs[i,j].set_yticklabels([])

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()


def create_basis(spin_base):
    """Returns the basis F_i using spin operators in
        represented in the basis 'spin_base'.

    e.g. spin_base = 'zx'
    means the spin opertors for the first spin are
    represented in the z-basis, whereas the spin
    operators for the second spin are represented in
    the x-basis.
    """
    paulis = {'z' : {'I' : qu.pauli('I'),
                    'X' : qu.pauli('X'),
                    'Y' : qu.pauli('Y'),
                    'Z' : qu.pauli('Z')},
              'y' : {'I' : qu.pauli('I'),
                    'X' : qu.pauli('Y'),  # X -> Y
                    'Y' : qu.pauli('Z'),  # Y -> Z
                    'Z' : qu.pauli('X')}, # Z -> X
              'x' : {'I' : qu.pauli('I'),
                    'X' : qu.pauli('Z'),  # X -> Z
                    'Y' : -qu.pauli('Y'), # Y -> -Y
                    'Z' : qu.pauli('X')}, # Z -> X
             }
    # the pauli op. in the right basis
    pauli1 = paulis[spin_base[0]]
    pauli2 = paulis[spin_base[1]]

    # build the F basis
    F = []
    for ob1, ob2 in product(['I', 'X', 'Y', 'Z'], repeat=2):
        F.append(0.5*pauli1[ob1]&pauli2[ob2])
    return F


def experimental_data(data, N):
    """The function reconstruct the rho, save the diagonal = probability.
    From the probability I can sample -> psi state.
    Having the state I can calculate the expecation value.

    Parameters
    ----------
    data : array
        1D array of tebd data (len == 15)
    N : int
        Number of samples to use

    """
    basis = ['xx', 'yy', 'zz', 'xy', 'xz', 'yx', 'yz', 'zx', 'zy']

    probability = {}
    for b in basis:
        F_b = create_basis(b)
        rho = np.matrix(np.eye(4, dtype=complex)/4) # the Identity
        for i in range(15):
            rho += data[i]*F_b[i+1]
        probability[b] = np.asarray(rho.diagonal().real)[0]

    samples = {}
    for key in probability.keys():
        unique, counts = np.unique(np.random.choice([0,1,2,3], size=N, p=probability[key]), return_counts=True)

        samples[key] = [0, 0, 0, 0]
        for u, c in zip(unique, counts):
            samples[key][u] = c/N

    # I need the operator of which calculate the expecation value: either I or sigma Z bacause
    # I put myself always in the right basis
    # Ans the basis in which calculate the expecation value
    operator = ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    basis    = ['xx', 'yy', 'zz', 'xx', 'xx', 'xy', 'xz', 'yy', 'yx', 'yy', 'yz', 'zz', 'zx', 'zy', 'zz']
    pauli = {'I' : qu.pauli('I').real,
            'X' : qu.pauli('Z').real,
            'Y' : qu.pauli('Z').real,
            'Z' : qu.pauli('Z').real}

    exp_data = [0]*15
    for i in range(15):
        psi = samples[basis[i]]
        op = pauli[operator[i][0]]&pauli[operator[i][1]]*0.5
        exp_data[i] = (op@psi).sum()

    return np.array(exp_data)
