def flatten(inp):
    return __flatten__(inp, [])
    
def __flatten__(inp, out : list):
    if type(inp) != type([]):
        out.append(inp)
        return out
    
    for element in inp:
        __flatten__(element, out)
    return out