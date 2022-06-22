
def simulate_storage(init_storage, net_inflow, release):
    storage = [0] * len(net_inflow)
    st = init_storage
    for i, (ni, rel) in enumerate(zip(net_inflow, release)):
        st = st + ni - rel
        storage[i] = st
    return storage
