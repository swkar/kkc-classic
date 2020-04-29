
# KKC-Classic solver for 2 and 3 dimensions
# Play it here: https://kkc-classic.pagehits.io/

import numpy as np
import itertools


def format_input(tup, dims=2):
    if isinstance(tup, (list, tuple)):
        return tuple(tup)
    if isinstance(tup, (int, np.int64)):
        return tuple(map(int, ('{:0'+str(dims)+'d}').format(tup)))
    return tup


def get_feedback(guess, actual=123, dims=None):
    if dims is None:
        dims = auto_width((actual, guess))

    actual, guess = (list(format_input(x, dims)) for x in (actual, guess))
    feedback = ''
    for n in reversed(range(dims)):
        if actual[n] == guess[n]:
            feedback += '+'
            _ = actual.pop(n), guess.pop(n)
    for g in guess:
        if g in actual:
            feedback += 'x'
            actual.remove(g)
    return feedback if feedback else '-'


def solve(actual=123, n_symbols=10, show=False, dims=None, seed=0, max_iter=100):
    if dims is None:
        dims = auto_width((actual,))

    actual = format_input(actual, dims=dims)
    player_class = [KKC2, KKC3][dims-2]
    np.random.seed(seed)
    player, n, answer = player_class(n_symbols=n_symbols, show=show), 0, None
    while True:
        n += 1
        guess = choice(player.prob)
        feedback = get_feedback(guess, actual, dims=dims)
        done = player.update(guess, feedback, show=show)
        if done or (n > max_iter):
            answer = choice(player.prob)
            if show:
                print('answer', '.'.join(map(str, answer)))
            break
    return answer, n


def replay(lst=(), show=True, player_class=None):
    if player_class is None:
        dims = auto_width(lst[::2])
        player_class = [KKC2, KKC3][dims-2]
        print('dims', dims)

    player = player_class(show=show)
    for n in range(0, len(lst), 2):
        guess, feedback = lst[n], lst[n+1]
        player.update(guess, feedback, show=show)
    return player


def auto_width(inputs):
    dims = np.max([len(format_input(x)) for x in inputs])
    return dims


def choice(prob):
    sample = np.random.choice(np.prod(prob.shape), p=prob.ravel())
    indices = np.unravel_index(sample, prob.shape)
    return list(map(int, indices))


def as_slice_list(ix_lst):
    # [1,2,4,9,10,22] --> [1:5, 9:11, 22:23]
    # 5 --> [5:6]
    if isinstance(ix_lst, int):
        ix_lst = [ix_lst]
    lst = []
    first, last = None, None
    for e in ix_lst:
        e = int(e)
        if first is None:
            first, last = e, e
            continue
        if last + 1 == e:
            last = e
            continue
        lst.append(slice(first, last+1))
        first, last = e, e

    if last is not None:
        lst.append(slice(first, last+1))
    return lst


def as_slices(all_indices):
    all_slices = []
    for indices in all_indices:
        slices = tuple(as_slice_list(ix_lst) for ix_lst in indices)
        all_slices += list(itertools.product(*slices))
    return all_slices


def compose(prob, indices, eps=1e-20):
    slices = as_slices(indices)
    acc = np.zeros_like(prob)
    for slc in slices:
        acc[slc] += prob[slc]

    acc /= np.max((np.sum(acc), eps))
    return acc


def sum_to_one(prob):
    return prob / np.sum(prob)


def entropy(prob, norm=False, eps=1e-20):
    if norm:
        prob = sum_to_one(prob)
    ent = -np.sum(prob * np.log(np.maximum(prob, eps)))
    return ent


class KKC3:
    def __init__(self, prob=None, n_symbols=10, show=False):
        if prob is None:
            prob = np.ones((n_symbols,)*3, dtype=np.float64)
        self.prob = prob

        prob[:, :] = sum_to_one(prob)
        if show:
            print('Init', n_symbols, *self.peek())

    def indices(self, exclude=()):
        symbols = set(range(self.prob.shape[0]))
        if isinstance(exclude, int):
            exclude = (exclude,)
        lst = sorted(symbols - set(exclude))
        return lst

    def update(self, guess, feedback, show=False):
        guess = format_input(guess, dims=3)
        feedback = ''.join(sorted(feedback, reverse=True))

        x, y, z = guess
        cx, cy, cz = self.indices(exclude=x), self.indices(exclude=y), self.indices(exclude=z)
        cxy, cyz, czx = self.indices(exclude=(x, y)), self.indices(exclude=(y, z)), self.indices(exclude=(z, x))
        cxyz = self.indices(exclude=(x, y, z))
        slices = ()

        if feedback == '-':
            slices = ((cxyz, cxyz, cxyz),)

        elif feedback == 'x':
            slices = ((cyz, x, cyz), (cyz, cyz, x),
                      (y, czx, czx), (czx, czx, y),
                      (z, cxy, cxy), (cxy, z, cxy))

        elif feedback == '+':
            slices = ((x, cyz, cyz), (czx, y, czx), (cxy, cxy, z))

        elif feedback == 'xx':
            slices = ((cx, z, y), (z, cx, y), (y, z, cx),
                      (cy, z, x), (z, cy, x), (z, x, cy),
                      (cz, x, y), (y, cz, x), (y, x, cz))

        elif feedback == 'x+':
            slices = ((y, cx, z), (z, y, cx),
                      (cy, x, z), (x, z, cy),
                      (cz, y, x), (x, cz, y))

        elif feedback == '++':
            slices = ((cx, y, z), (x, cy, z), (x, y, cz))

        elif feedback == 'xxx':
            slices = ((z, x, y), (y, z, x))

        elif feedback == 'xx+':
            slices = ((x, z, y), (z, y, x), (y, x, z))

        elif feedback == '+++':
            slices = ((x, y, z),)

        self.prob = compose(self.prob, slices)

        if show:
            print('.'.join(map(str, guess))+', '+feedback, '-->', *self.peek())

        done = np.isclose(entropy(self.prob), 0)
        return done

    @staticmethod
    def format(p):
        if p == 0: return ''
        if p == 1: return '1'
        return '{:6.4f}'.format(p)

    def peek(self):
        ent = entropy(self.prob)
        ret = ['ent {:5.2f}'.format(ent)]
        if np.isclose(ent, 0): ret.append('BINGO!')
        return ret


class KKC2:
    def __init__(self, prob=None, n_symbols=10, show=False):
        if prob is None:
            prob = np.ones((n_symbols,)*2, dtype=np.float64)
        self.prob = prob

        prob[:, :] = sum_to_one(prob)
        if show:
            print('Init', n_symbols, *self.peek())

    def indices(self, exclude=()):
        symbols = set(range(self.prob.shape[0]))
        if isinstance(exclude, int):
            exclude = (exclude,)
        return sorted(symbols - set(exclude))

    def update(self, guess, feedback, show=False):
        guess = format_input(guess, dims=2)
        feedback = ''.join(sorted(feedback, reverse=True))

        x, y = guess
        cx, cy = self.indices(exclude=x), self.indices(exclude=y)
        cxy = self.indices(exclude=(x, y))
        slices = ()

        if feedback == '-':
            slices = ((cxy, cxy),)

        elif feedback == '++':
            slices = ((x, y),)

        elif feedback == '+':
            slices = ((x, cy), (cx, y))

        elif feedback == 'x':
            slices = ((y, cx), (cy, x))

        elif feedback == 'xx':
            slices = ((y, x),)

        self.prob = compose(self.prob, slices)

        if show:
            #self.print_state()
            print('.'.join(map(str, guess))+', '+feedback, '-->', *self.peek())

        done = np.isclose(entropy(self.prob), 0)
        return done

    @staticmethod
    def format(p):
        if p == 0: return ''
        if p == 1: return '1'
        return '{:6.4f}'.format(p)

    def print_state(self):
        n_symbols, prob = self.prob.shape[0], self.prob
        symbols = ['{:^6d}'.format(n) for n in range(n_symbols)]
        print(('{:^6}' + '|{:^6}' * n_symbols).format('x\\y', *symbols))
        for x in range(n_symbols):
            desc = [self.format(prob[x, y]) for y in range(n_symbols)]
            print(('{:^6}' + '|{:^6}' * n_symbols).format(symbols[x],  *desc))
        return

    def peek(self):
        ent = entropy(self.prob)
        ret = ['ent {:5.2f}'.format(ent)]
        if np.isclose(ent, 0): ret.append('BINGO!')
        return ret

