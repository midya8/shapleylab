"""Original supplied synthetic controls for the two-level allocation audit."""
import copy


def _request(names,parts,terms,point=None,background=None,bias=0,output='raw'):
    return {'feature_names':names,'point':point if point is not None else [1]*len(names),
        'background':background if background is not None else [[0]*len(names)],
        'groups':[{'name':'block '+str(k+1),'features':row}for k,row in enumerate(parts)],
        'model':{'bias':bias,'terms':[{'features':support,'weight':weight}for support,weight in terms],'output':output}}


_CONTROLS=[
    {'id':'three-way-union','title':'One interaction, three allocations','description':'A three-way interaction is shared between two blocks first, then between members. Ordinary feature credit and block sums ask different questions.',
     'request':_request(list('abc'),[[0,1],[2]],[([0,1,2],1)])},
    {'id':'singleton-partition','title':'Every feature is its own block','description':'Singleton blocks remove the ordering constraint and recover ordinary individual Shapley.',
     'request':_request(list('abc'),[[0],[1],[2]],[([0,1,2],1)])},
    {'id':'grand-union','title':'Only one block','description':'A single grand block also permits every internal ordering. Group credit is the entire prediction change, not three separate block players.',
     'request':_request(list('abc'),[[0,1,2]],[([0,1,2],1)])},
    {'id':'unequal-members','title':'A block is not an equal split','description':'Linear member effects have different sizes. Equal division of block credit would erase them.',
     'request':_request(list('abc'),[[0,1],[2]],[([0],2),([1],1),([2],3)])},
    {'id':'binary-hybrid-cancellation','title':'Zero credit can hide off-reference hybrids','description':'The binary disagreement score a+b-2ab is zero on both diagonal reference rows and the target. Partial within-block replacement still visits nonzero off-diagonal hybrids. Cancellation is not a manifold certificate.',
     'request':_request(list('ab'),[[0,1]],[([0],1),([1],1),([0,1],-2)],background=[[0,0],[1,1]])},
    {'id':'joint-background','title':'Background rows stay joint','description':'Missing coordinates come from one complete background row. Owen changes allowed feature orderings, not the declared replacement distribution.',
     'request':_request(list('abc'),[[0,1],[2]],[([0,1],1)],point=[2,3,0],background=[[0,0,0],[1,1,1]])},
    {'id':'negative-interaction','title':'Negative credit is still credit','description':'A negative three-way term has the same admissible-order weights, with signed marginal effects.',
     'request':_request(list('abc'),[[0,1],[2]],[([0,1,2],-4)],bias=7)},
    {'id':'mixed-effects','title':'Interactions and direct effects coexist','description':'No post-hoc normalization or equal within-block split is applied to mixed direct, pair and three-way effects.',
     'request':_request(list('abcd'),[[0,2],[1,3]],[([0],2),([1],-1),([0,1],3),([0,1,2],4),([2,3],-2)],bias=5)},
    {'id':'dummy-in-union','title':'A dummy can sit inside a block','description':'The fourth coordinate does not enter the score. It remains a member of the declared partition but its every marginal is zero.',
     'request':_request(list('abcd'),[[0,1,3],[2]],[([0,1,2],1)])},
    {'id':'constant-model','title':'No change to allocate','description':'A constant nonzero baseline gives zero member and block credits. The baseline itself is not distributed as a feature effect.',
     'request':_request(list('abc'),[[2,0],[1]],[],bias=9)},
    {'id':'two-unequal-unions','title':'Unequal block sizes','description':'A four-way interaction touches two blocks. Member weights depend on the active members inside each block, not the number of raw features alone.',
     'request':_request(list('abcd'),[[0,1,2],[3]],[([0,1,2,3],1)])},
    {'id':'six-feature-sigmoid','title':'A declared nonlinear output link','description':'A fixed supplied multiaffine score passes through a sigmoid. These are explanation arithmetic controls, not trained predictive performance.',
     'request':_request(list('abcdef'),[[0,3],[1,4],[2,5]],[([0],1),([2],-1),([1,3],2),([0,2,4],-2),([1,2,5],1)],point=[1,-1,2,0,1,-2],background=[[0]*6,[1]*6],bias=-1,output='sigmoid')}
]


def definitions():return copy.deepcopy(_CONTROLS)
