"""The original seven exact development fixtures, with no fitted model."""
import copy
from .model import polynomial
from .ledger import audit
from .demo import run_demo


def definitions():
    return {
      'product_zero_background':dict(label='Product / zero background',kind='product',point=[2,3,99],background=[[0,0,0],[0,0,9]],feature_names=['a','b','unused']),
      'product_shifted_background':dict(label='Product / shifted background',kind='product',point=[2,3,99],background=[[1,1,0],[1,1,9]],feature_names=['a','b','unused']),
      'joint_background_dependence':dict(label='Product / joint binary background',kind='product',point=[1,1],background=[[0,0],[1,1]],feature_names=['a','b']),
      'off_manifold_hybrids':dict(label='Squared difference / diagonal background',kind='squared_difference',point=[1,1],background=[[0,0],[1,1]],feature_names=['a','b']),
      'sigmoid_polynomial':dict(label='Sigmoid polynomial / three rows',kind='sigmoid',point=[1.5,.5,7],background=[[0,0,0],[1,1,1],[-1,.5,2]],feature_names=['a','b','unused']),
      'three_way_individual':dict(label='Three-way product / individual players',kind='triple',point=[1,1,1],background=[[0,0,0]],feature_names=['a','b','c']),
      'three_way_grouped':dict(label='Three-way product / atomic ab and c',kind='triple',point=[1,1,1],background=[[0,0,0]],feature_names=['a','b','c'],groups=[dict(name='ab',features=[0,1]),dict(name='c',features=[2])]),
    }


def scorer(kind):
    if kind=='product':return lambda row:row[0]*row[1]
    if kind=='squared_difference':return lambda row:(row[0]-row[1])**2
    if kind=='triple':return lambda row:row[0]*row[1]*row[2]
    if kind=='sigmoid':return polynomial(dict(bias=-.3,linear=[1.2,-.8,0.],pairs=[dict(i=0,j=1,weight=.7)],output='sigmoid'),3)
    raise ValueError('unknown fixed synthetic score')


def inspect(name, point=None):
    arms=definitions()
    if not isinstance(name,str) or name not in arms:raise ValueError('unknown fixture')
    a=arms[name]
    x=copy.deepcopy(a['point'] if point is None else point)
    return audit(scorer(a['kind']),x,a['background'],a['feature_names'],a.get('groups'))


def build():
    arms={}
    for name,a in definitions().items():
        edits=[]
        for j in range(len(a['point'])):
            for value in [a['point'][j],-1,0,1,2]:
                x=a['point'][:];x[j]=value
                edits.append(dict(feature=j,value=value,witness=inspect(name,x)))
        arms[name]=dict(definition=a,baseline=inspect(name),edits=edits)
    return dict(version=1,arms=arms,original_demo=run_demo())
