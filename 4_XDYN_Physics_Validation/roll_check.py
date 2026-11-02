import csv, math
r = list(csv.reader(open('waves.csv')))
h = r[0]
i = h.index('phi(dtmb)')
t = [float(x[0]) for x in r[1:]]
p = [float(x[i]) for x in r[1:]]
w = [(a, b) for a, b in zip(t, p) if a >= 100]
m = sum(b for _, b in w) / len(w)
u = [w[k][0] for k in range(1, len(w)) if w[k-1][1] - m < 0 <= w[k][1] - m]
period = (u[-1] - u[0]) / (len(u) - 1)
early = math.degrees(max(abs(b) for a, b in zip(t, p) if a < 60))
late = math.degrees(max(abs(b) for a, b in zip(t, p) if a > 240))
print('roll period %.1f s, early roll %.2f deg, late roll %.2f deg' % (period, early, late))
