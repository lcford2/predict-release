import numpy as np

def find_leaf(rel_pre):
    """
    Take the standardized value of previous release (rel_pre) and return what leaf this observation falls in.
    """
    if rel_pre <= -0.879:
        return 0
    elif rel_pre <= -0.5:
        return 1
    elif rel_pre <= -0.027:
        return 2
    elif rel_pre <= 0.423:
        return 3
    elif rel_pre <= 1.034:
        return 4
    elif rel_pre <= 1.8:
        return 5
    elif rel_pre <= 2.783:
        return 6
    else:
        return 7
    
def get_params(leaf):
    """
    Return the parameter values for leaf in the following order:
    sto_diff, s x i, rel pre, rel roll, inf, inf roll
    """
    params = [
        [1.7830471551933158, 0.081428743966359, 0.5604664716660999, 0.4448480434109587, -0.0710651774099053, -0.0947637517087667],
        [1.9963262711875127, 0.1075011350215922, 0.5819375579681747, 0.4170508929513588, -0.0315444958669184, -0.1689942561437014],
        [2.8637551195578936, 0.1171903066597002, 0.6222110496636013, 0.4031452925245382, -0.0861928844796358, -0.1482699797743603],
        [2.934834020479741, 0.0880844542989738, 0.569788512685431, 0.369139177153702, -0.0561892434371325, -0.1504614334074467],
        [2.96417519725766, 0.1241271210917402, 0.6265170607100643, 0.3127217845042764, -0.0925619939003157, -0.1325097773669954],
        [2.621741873238648, 0.0621911010093336, 0.6421504380913845, 0.2757786613436721, -0.0575031396248832, -0.0593016973917626],
        [1.2967599933944882, 0.0197643821493871, 0.6543769985239649, 0.1956802260296211, -0.0641736418134653, 0.0859421560903332],
        [1.1836888244344834, 0.104833235519786, 0.7488969225356108, 0.1368789879895899, -0.0986047149291255, 0.0506763541416042]
    ]
    return params[leaf]

vec_find_leaf = np.vectorize(find_leaf)

def run_model(X):
    """
    This function gets the leaves and the corresponding parameters for an X matrix of independent variables.
    IMPORTANT: this function requires data be input in the following order:
    sto_diff, s x i, rel pre, rel roll, inf, inf roll
    """
    leaves = vec_find_leaf(X[:,2])
    params = np.array([get_params(l) for l in leaves])
    return (X*params).sum(axis=1)        
    
if __name__ == "__main__":
    from IPython import embed as II
    II()