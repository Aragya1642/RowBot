import csv, math
r = list(csv.reader(open('waves.csv')))
h = r[0]
t = [float(x[0]) for x in r[1:]]
psi = [math.degrees(float(x[h.index('psi(dtmb)')])) for x in r[1:]]
y = [float(x[h.index('y(dtmb)')]) for x in r[1:]]
for target in [0, 60, 120, 180, 240, 300]:
    k = min(range(len(t)), key=lambda j: abs(t[j] - target))
    print('t=%4.0f s  heading %7.1f deg  sideways drift y %7.1f m' % (t[k], psi[k], y[k]))
