"""Reconstruye los cinco QR desde sus URL y los compara módulo a módulo con los de data/hopfield.

Con la reconstrucción se sabe qué bit del mensaje ocupa cada módulo: cuántos de los módulos que
difieren entre los QR son datos y cuántos corrección de errores, y en cuántos bits de datos difiere
cada par. Implementa sólo lo que usan estos QR: versión 5 (37×37), modo byte, un bloque.

Uso desde TP4: python3 analysis/qr_bits.py
"""
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis.hopfield_qr import QRS as QR_FILES, load_patterns

QRS = list(QR_FILES)

PREFIX = 'https://github.com/eugepineiro/sia-tips/blob/feature/tp4-images/TPs/TP4/'
FILES = {'menu_principal': 'food', 'vinos': 'wine', 'cocteles_y_bebidas': 'beverages',
         'postres': 'dessert', 'especiales': 'specials'}
SIZE = 37
ECL_BITS = {'L': 1, 'M': 0, 'Q': 3, 'H': 2}
ECC_PER_LEVEL = {'L': (108, 26), 'M': (86, 24), 'Q': (62, 18), 'H': (46, 22)}  # v5: (datos, ecc por bloque)


def format_bits(ecl, mask):
    data = ECL_BITS[ecl] << 3 | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return (data << 10 | rem) ^ 0x5412


def gf_mul(x, y):
    z = 0
    for i in reversed(range(8)):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def rs_divisor(degree):
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(len(result)):
            result[j] = gf_mul(result[j], root)
            if j + 1 < len(result):
                result[j] ^= result[j + 1]
        root = gf_mul(root, 0x02)
    return result


def rs_remainder(data, divisor):
    result = [0] * len(divisor)
    for b in data:
        factor = b ^ result.pop(0)
        result.append(0)
        for i, coef in enumerate(divisor):
            result[i] ^= gf_mul(coef, factor)
    return result


def data_codewords(text, capacity):
    bits = '0100' + format(len(text), '08b') + ''.join(format(b, '08b') for b in text.encode())
    bits += '0' * min(4, capacity * 8 - len(bits))
    bits += '0' * (-len(bits) % 8)
    words = [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(words) < capacity:
        words.append(pad)
        pad ^= 0xEC ^ 0x11
    return words


def function_modules():
    func = np.zeros((SIZE, SIZE), bool)
    mod = np.zeros((SIZE, SIZE), bool)

    def put(x, y, dark):
        mod[y, x], func[y, x] = dark, True
    for i in range(SIZE):
        put(6, i, i % 2 == 0)
        put(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (SIZE - 4, 3), (3, SIZE - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < SIZE and 0 <= y < SIZE:
                    dist = max(abs(dx), abs(dy))
                    put(x, y, dist not in (2, 4))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            put(30 + dx, 30 + dy, max(abs(dx), abs(dy)) != 1)
    return func, mod


def draw_format(func, mod, bits):
    def put(x, y, dark):
        mod[y, x], func[y, x] = dark, True
    bit = lambda i: (bits >> i) & 1 == 1  # noqa: E731
    for i in range(6):
        put(8, i, bit(i))
    put(8, 7, bit(6)); put(8, 8, bit(7)); put(7, 8, bit(8))
    for i in range(9, 15):
        put(14 - i, 8, bit(i))
    for i in range(8):
        put(SIZE - 1 - i, 8, bit(i))
    for i in range(8, 15):
        put(8, SIZE - 15 + i, bit(i))
    put(8, SIZE - 8, True)


MASKS = [lambda x, y: (x + y) % 2 == 0, lambda x, y: y % 2 == 0, lambda x, y: x % 3 == 0,
         lambda x, y: (x + y) % 3 == 0, lambda x, y: (x // 3 + y // 2) % 2 == 0,
         lambda x, y: x * y % 2 + x * y % 3 == 0, lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
         lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0]


def build(text, ecl, mask):
    """Devuelve la matriz y, para cada módulo de datos, el índice del bit del flujo que lo ocupa (-1 si no)."""
    data_cap, ecc = ECC_PER_LEVEL[ecl]
    words = data_codewords(text, data_cap)
    words += rs_remainder(words, rs_divisor(ecc))
    func, mod = function_modules()
    draw_format(func, mod, format_bits(ecl, mask))
    owner = -np.ones((SIZE, SIZE), int)
    i = 0
    right = SIZE - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(SIZE):
            for j in range(2):
                x = right - j
                upward = ((right + 1) & 2) == 0
                y = SIZE - 1 - vert if upward else vert
                if not func[y, x] and i < len(words) * 8:
                    mod[y, x] = (words[i >> 3] >> (7 - (i & 7))) & 1 == 1
                    owner[y, x] = i
                    i += 1
        right -= 2
    for y in range(SIZE):
        for x in range(SIZE):
            if not func[y, x] and MASKS[mask](x, y):
                mod[y, x] = not mod[y, x]
    return mod, owner, words


def bits(words):
    return np.array([int(bit) for word in words for bit in format(word, '08b')])


def main():
    patterns = load_patterns().reshape(-1, SIZE, SIZE) > 0  # módulo negro = +1
    wrong, ecl, mask = min((int(np.sum(build(PREFIX + FILES[QRS[0]] + '_menu.png', ecl, mask)[0] != patterns[0])),
                            ecl, mask) for ecl in ECL_BITS for mask in range(8))
    print(f'formato: corrección {ecl}, máscara {mask}')
    built = {}
    for k, name in enumerate(QRS):
        text = PREFIX + FILES[name] + '_menu.png'
        built[name] = build(text, ecl, mask)
        print(f'{name:20s} {len(text)} caracteres, módulos distintos de la imagen: '
              f'{int(np.sum(built[name][0] != patterns[k]))}')

    data_bits = ECC_PER_LEVEL[ecl][0] * 8
    stacked = np.stack([built[name][0] for name in QRS])
    differing = (stacked != stacked[0]).any(0)
    owner = built[QRS[0]][1][differing]
    print(f'\nmódulos que difieren: {int(differing.sum())} = {int(np.sum(owner < data_bits))} de datos'
          f' + {int(np.sum(owner >= data_bits))} de corrección de errores')
    print('\npar, bits de datos distintos, bits de corrección distintos, solapamiento sin lo común')
    for a, b in combinations(QRS, 2):
        da, db = bits(built[a][2]), bits(built[b][2])
        overlap = np.mean(np.where(built[a][0][differing] == built[b][0][differing], 1, -1))
        print(f'{a:20s} {b:20s} {int(np.sum(da[:data_bits] != db[:data_bits])):4d}/{data_bits}'
              f' {int(np.sum(da[data_bits:] != db[data_bits:])):4d}/{len(da) - data_bits} {overlap:7.3f}')


if __name__ == '__main__':
    main()
