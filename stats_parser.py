import os
import re
import sys

COLUMNS = [
    'EXPLORATION_TIME', 'MINIBATCH_SIZE', 'CONSTANT_LEARNING_RATE',
    'START_LEARNING_RATE', 'FINAL_LEARNING_RATE', 'LEARNING_RATE_CHANGE_EPISODES',
    'NETWORK_ARCHITECTURE', 'DISCOUNT_FACTOR', 'CLIP_DELTA', 'SWAP_NETWORKS_EVERY',
    'FINAL_EPSILON_GREEDY', 'USE_LASAGNE', 'simulator time [h]',
    'result (avg) [s]', '\'+-', 'result (median) [s]',
]


def float_round(f, digits=4, multiply=1.0):
    return str(round(float(f) * multiply, digits))

print('\t'.join(COLUMNS))

for d in os.listdir(sys.argv[-1]):
    d = os.path.abspath(os.path.join(sys.argv[-1], d))
    if not os.path.isdir(d):
        continue
    # print('Processing {}'.format(d))
    l = {}
    for i, line in enumerate(open(os.path.join(d, 'stats.txt'))):
        line = line.strip()
        if i == 1:
            r = line.strip().split(':')[-1]
            result, plus_minus = re.search(r'([\d.]+) \(\+- ([\d.]+)\)', r).groups()
            l['result (avg) [s]'] = float_round(result)
            l['\'+-'] = float_round(plus_minus)
        elif i == 3:
            l['simulator time [h]'] = float_round(
                line.strip().split(':')[-1],
                multiply=(1.0 / (60 * 60)),
                digits=1
            )
        elif i == 4:
            l['result (median) [s]'] = float_round(line.strip().split(':')[-1])
        else:
            s = line.split('=')
            if s[0] in COLUMNS:
                l[s[0]] = s[1]
    for c in COLUMNS:
        print('{}\t'.format(l.get(c, '-')), end="")
    print('')
