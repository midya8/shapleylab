from .core import explain
from .grouped import explain_groups
from .model import polynomial


def run_demo():
    product=explain(lambda x:x[0]*x[1],[2,3,99],[[0,0,0],[0,0,9]],['a','b','unused'])
    shifted=explain(lambda x:x[0]*x[1],[2,3,99],[[1,1,0],[1,1,9]],['a','b','unused'])
    dependent=explain(lambda x:x[0]*x[1],[1,1],[[0,0],[1,1]],['a','b'])
    offmanifold=explain(lambda x:(x[0]-x[1])**2,[1,1],[[0,0],[1,1]],['a','b'])
    assert offmanifold['coalition_values']==[0.,.5,.5,0.]
    score=polynomial({'bias':-.3,'linear':[1.2,-.8,0.],
                      'pairs':[{'i':0,'j':1,'weight':.7}],'output':'sigmoid'},3)
    nonlinear=explain(score,[1.5,.5,7],[[0,0,0],[1,1,1],[-1,.5,2]],['a','b','unused'])
    assert product['attributions']==[3.,3.,0.]
    assert product['model_evaluations']==15
    assert product['prediction']==shifted['prediction']==6
    assert product['attributions']!=shifted['attributions']
    assert dependent['baseline']==.5  # Not .25: missing features remain jointly sampled.
    assert abs(nonlinear['efficiency_residual'])<1e-12
    triple=lambda x:x[0]*x[1]*x[2]
    individual=explain(triple,[1,1,1],[[0,0,0]],['a','b','c'])
    grouped=explain_groups(triple,[1,1,1],[[0,0,0]],['a','b','c'],
                           [{'name':'ab','features':[0,1]},{'name':'c','features':[2]}])
    assert grouped['attributions']==[.5,.5]
    assert individual['attributions']==[1/3,1/3,1/3]
    assert sum(individual['attributions'][:2])!=grouped['attributions'][0]
    return {'three_way_individual':individual,'three_way_grouped':grouped,
            'product_zero_background':product,'product_shifted_background':shifted,
            'joint_background_dependence':dependent,'off_manifold_hybrids':offmanifold,
            'sigmoid_polynomial':nonlinear,
            'scope':'Selected hand-constructed arithmetic fixtures, not a trained-model quality benchmark. '
                    'Exact means enumerate every coalition for the declared finite-background game, '
                    'not conditional/causal correctness. Same prediction can have different attributions.'}
