
def mean(data, key = "total_bill"):
    values = [float(row[key]) for row in data]
    return(sum(values)/len(values) if values else 0)

def variance(data, key = "total_bill", ddof = 1): 
    values = [float(row[key]) for row in data]
    n = len(values)
    if n == 0: return 0
    mean_x = sum(values) / n
    squared = [(x-mean_x) ** 2 for x in values]
    return sum(squared) / (n - ddof)

def std(data, key = "total_bill", ddof = 1):
    var = variance(data, key, ddof)
    return var ** 0.5

def median(data, key = "total_bill"):
    values = [float(row[key]) for row in data]
    n = len(values)
    values = sorted(values)
    if n%2==1:
        return values[n//2]
    else:
        return (values[n//2 - 1] + values[n//2]) / 2

def mode(data, key = "total_bill"):
    counter = {}
    for row in data:
        val = row[key]
        counter[val] = counter.get(val, 0) + 1

    max_count = max(counter.values())
    modes = [key for key, value in counter.items() if value == max_count]

    return modes, max_count

def cov(data, x = "total_bill", y = "tip", ddof=1):
    values_x = [float(row[x]) for row in data]
    values_y = [float(row[y]) for row in data]
    n = len(data)
    mean_x = sum(values_x) / n
    mean_y = sum(values_y) / n
    sum_cross = sum((values_x[i] - mean_x) * (values_y[i] - mean_y) for i in range(n))

    return sum_cross / (n - ddof)

def cor(data, x = "total_bill", y = "tip"):
    cov_xy = cov(data, x, y)
    return cov_xy / (std(data, x) * std(data, y))

def empirical_probability(data, predicate):
    n = len(data)
    if n == 0:
        raise ValueError("Пустая выборка")
    favorable = sum(1 for row in data if predicate(row))
    return favorable / n

def conditional_probability(data, event_pred, cond_pred):
    condit_pred = [row for row in data if cond_pred(row)]
    n = len(condit_pred)
    if n == 0: return "В выборке нет строк, удовлетворяющих условию B"
    sum_event = sum(1 for row in condit_pred if event_pred(row))
    return sum_event / n

def distribution_table(data, key = "sex"):
    n = len(data)
    counter = {}
    for row in data:
        val = row[key]
        counter[val] = counter.get(val, 0) + 1
    return {k: value/n for k, value in sorted(counter.items()) }