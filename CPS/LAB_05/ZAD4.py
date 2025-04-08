import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (
    buttap, cheb1ap, cheb2ap, ellipap,
    lp2lp, lp2hp, lp2bp, lp2bs,
    freqs, impulse, step, lti
)

N = 8
f0 = 100
f1, f2 = 10, 100
Rp = 3
Rs = 100

prototypes = [
    (buttap(N), 'Butterworth'),
    (cheb1ap(N, Rp), 'Chebyshev I'),
    (cheb2ap(N, Rs), 'Chebyshev II'),
    (ellipap(N, Rp, Rs), 'Elliptic')
]

'''
WYDAJE MI SIĘ ŻE STĄD TRZEBA WYBRAĆ TYLKO JEDEN RODZAJ FILTRU, ALE DLA PEWNOŚCI DAŁEM WSZYSTKIE - ewentualnie pozniej zmienie kod zeby tylko do jednego robił przejscie bo tak to generuje w chuj wykresow
'''

transformations = [
    ('LP', lambda b, a: lp2lp(b, a, 2 * np.pi * f0)),
    ('HP', lambda b, a: lp2hp(b, a, 2 * np.pi * f0)),
    ('BP', lambda b, a: lp2bp(b, a, 2 * np.pi * np.sqrt(f1 * f2), 2 * np.pi * (f2 - f1))),
    ('BS', lambda b, a: lp2bs(b, a, 2 * np.pi * np.sqrt(f1 * f2), 2 * np.pi * (f2 - f1)))
]


for (proto, proto_name) in prototypes:
    z, p, k = proto
    b = np.atleast_1d(k * np.poly(z))
    a = np.atleast_1d(np.poly(p))

    f = np.arange(0.01, 1000.01, 0.01)
    w = 2 * np.pi * f
    _, H_proto = freqs(b, a, w)

    plt.figure()
    plt.semilogx(w, 20 * np.log10(np.abs(H_proto)))
    plt.title(f'{proto_name} - Prototyp |H(f)|')
    plt.xlabel('w [rad/s]')
    plt.ylabel('|H(f)| [dB]')
    plt.grid(True)

    for (trans_name, trans_func) in transformations:
        b_t, a_t = trans_func(b, a)

        z_t = np.roots(b_t)
        p_t = np.roots(a_t)

        _, H_t = freqs(b_t, a_t, w)

        plt.figure()
        plt.semilogx(w, 20 * np.log10(np.abs(H_t)))
        plt.title(f'Ch. amplitudowa: {proto_name} -> {trans_name} |H(f)|')
        plt.xlabel('w [rad/s]')
        plt.ylabel('|H(f)| [dB]')
        plt.grid(True)

        plt.figure()
        plt.plot(np.real(z_t), np.imag(z_t), 'ro', label='Zera')
        plt.plot(np.real(p_t), np.imag(p_t), 'b*', label='Bieguny')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.title(f'{proto_name} -> {trans_name} Zera i Bieguny')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()

        system = lti(b_t, a_t)
        t_imp, y_imp = impulse(system)
        t_step, y_step = step(system)

        plt.figure()
        plt.plot(t_imp, y_imp)
        plt.title(f'{proto_name} -> {trans_name} Odpowiedź Impulsowa')
        plt.grid(True)

        plt.figure()
        plt.plot(t_step, y_step)
        plt.title(f'{proto_name} -> {trans_name} Odpowiedź Skokowa')
        plt.grid(True)

plt.show()