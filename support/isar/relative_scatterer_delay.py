"""Test the relative envelope delay of a common pair at co-sited transmitters.

For the same two physical points, their excess-path difference is identical
in bands radiated from the same transmitter phase centre. The fixed digital
range origin, receiver timing and transmitter direct-path reference cancel
within each band. Constant carrier phases remain independent in this control.
No observation is shifted. Actual per-channel antenna offsets belong in the
later exact geometric model; this is a nominal co-sited hypothesis test.
"""
import numpy as np

from physical_pair import fit_parameters


class RelativeScattererDelay:
    def __init__(self,base,first,second,groups):
        n=base.components;nb=len(base.bands)
        if first==second or not 0<=first<n or not 0<=second<n:raise ValueError('Two distinct components required')
        groups=[tuple(g) for g in groups]
        members=[c for g in groups for c in g]
        if sorted(members)!=list(range(nb)) or any(not g for g in groups):
            raise ValueError('Disjoint nonempty groups must cover every band exactly once')
        self.base=base;self.groups=groups;self.first=first;self.second=second
        self.origins=np.array([[2*n*c+2*s for s in (first,second)] for c in range(nb)])
        removed=[self.origins[c,1] for group in groups for c in group[1:]]
        self.kept=np.setdiff1d(np.arange(base.count),removed);self.count=len(self.kept);self.energy=base.energy
        transform=np.zeros((base.count,self.count));transform[self.kept,np.arange(self.count)]=1.
        for group in groups:
            ref=group[0]
            for c in group[1:]:
                a,b=self.origins[c];ra,rb=self.origins[ref]
                transform[b]=transform[a]+transform[rb]-transform[ra]
        self.transform=transform;self.lower=base.lower[self.kept];self.upper=base.upper[self.kept];self.last=None

    def mapping(self,parameters):return self.transform@np.asarray(parameters)

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residual,jacobian=self.base.evaluate(self.mapping(parameters));jacobian=jacobian@self.transform
        self.last=(np.asarray(parameters).copy(),residual,jacobian)
        return residual,jacobian

    def objective(self,parameters):
        residual,jacobian=self.evaluate(parameters)
        return float(residual@residual),2*jacobian.T@residual

    def guard_audit(self,parameters):
        """Removed envelope guards are checked explicitly, never virtual data.

        Least-squares only bounds the retained coordinates. A result outside
        any original guard is an invalid local control, not accepted evidence.
        """
        p=self.mapping(parameters)
        violation=np.maximum(np.maximum(self.base.lower-p,p-self.base.upper),0.)
        return dict(maximum_violation=float(violation.max(initial=0.)),
            outside_indices=np.flatnonzero(violation>1e-7).tolist(),valid=bool(np.all(violation<=1e-7)))

    def fit(self,parameters,*,frozen_base_indices=(),max_evaluations=220):
        free=np.flatnonzero(~np.isin(self.kept,frozen_base_indices))
        q,audit=fit_parameters(self,parameters,free,max_evaluations=max_evaluations)
        audit['mapped_envelope_guard']=self.guard_audit(q)
        return q,audit


if __name__=='__main__':raise SystemExit('Import this nested hypothesis from a recorded-data control.')
